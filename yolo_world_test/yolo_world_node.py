import os
import json
import time
import base64
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import supervision as sv
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import SAM as UltraSAM

try:
    # Optional, for better NMS when ONNX is exported without NMS
    from torchvision.ops import nms, batched_nms
except Exception:
    nms = None
    batched_nms = None

from scene_graph.utils.config import build_config


class YoloWorldOnnxNode(Node):
    """
    ROS2 node:
      - subscribes to RGB image topic
      - runs YOLO-World ONNX (/data/yolo-world_custom_5cls.onnx) via ONNXRuntime GPU
      - runs mobile-SAM for masks
      - publishes annotated image
    """

    def __init__(self) -> None:
        super().__init__("yolo_sam_node")

        self.get_logger().info("[YOLO-ONNX] Building config...")
        cfg = build_config()
        self.cfg = cfg

        image_topic = self.declare_parameter(
            "image_topic", cfg.image_topic
        ).get_parameter_value().string_value
        viz_topic = self.declare_parameter(
            "visualization_topic", "visualization/test"
        ).get_parameter_value().string_value
        det_topic = self.declare_parameter(
            "detections_topic", "yolo_sam/detections"
        ).get_parameter_value().string_value

        self.get_logger().info(
            f"[YOLO-ONNX] Subscribing image from: {image_topic}, "
            f"publishing visualization to: {viz_topic}, detections(+segments) to: {det_topic}"
        )

        self.bridge = CvBridge()

        # Model/resource paths and detector options
        onnx_model_path = self.declare_parameter(
            "onnx_model_path", "/data/yolo-world_custom_5cls.onnx"
        ).get_parameter_value().string_value
        sam_weight_path = self.declare_parameter(
            "sam_weights", str(getattr(cfg, "sam_weights", "mobile_sam.pt") or "mobile_sam.pt")
        ).get_parameter_value().string_value
        object_list = self.declare_parameter(
            "object_list", str(getattr(cfg, "object_classes", "sign,bench,car,building,tower,firehydrant"))
        ).get_parameter_value().string_value

        # YOLO thresholds from config (used when ONNX has no NMS)
        self.yolo_score_thr = float(getattr(cfg, "yolo_conf", 0.30))
        self.yolo_topk = int(getattr(cfg, "yolo_topk", 100))

        # ------------------------------------------------------------------
        # Prepare YOLO-World class texts from config
        # ------------------------------------------------------------------
        self.yolo_texts: List[List[str]] = []
        try:
            obj_classes = object_list
            if isinstance(obj_classes, str) and obj_classes.endswith(".txt"):
                with open(obj_classes) as f:
                    lines = f.readlines()
                self.yolo_texts = [[t.rstrip("\r\n")] for t in lines]
            elif isinstance(obj_classes, str):
                self.yolo_texts = [[t.strip()] for t in obj_classes.split(",")]
            else:
                self.yolo_texts = [[str(t)] for t in list(obj_classes)]
        except Exception as e:
            self.get_logger().warning(
                f"[YOLO-ONNX] Failed to parse object_classes: {e}"
            )
            self.yolo_texts = []

        # ------------------------------------------------------------------
        # Initialize SAM (mobile-SAM) for segmentation on detections
        # ------------------------------------------------------------------
        self.sam_model = None
        self.sam_conf = float(getattr(cfg, "sam_conf", 0.30))
        try:
            sam_device = (
                "cuda:0"
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else "cpu"
            )
            self.get_logger().info(
                f"[YOLO-ONNX] Initializing mobile-SAM with weights: {sam_weight_path} on {sam_device}"
            )
            try:
                self.sam_model = UltraSAM(sam_weight_path).to(sam_device)
            except Exception as e:
                self.get_logger().warning(
                    f"[YOLO-ONNX] Failed to load mobile-SAM weights '{sam_weight_path}': {e}"
                )
                try:
                    fallback_weights = "mobile_sam.pt"
                    self.sam_model = UltraSAM(fallback_weights).to(sam_device)
                    self.get_logger().info(
                        f"[YOLO-ONNX] Fallback mobile-SAM model '{fallback_weights}' loaded."
                    )
                except Exception as e2:
                    self.sam_model = None
                    self.get_logger().warning(
                        f"[YOLO-ONNX] mobile-SAM initialization failed completely: {e2}"
                    )

            if self.sam_model is not None:
                try:
                    self.sam_model.eval()
                except Exception:
                    pass
                self.get_logger().info(
                    "[YOLO-ONNX] mobile-SAM model initialized successfully."
                )
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] mobile-SAM setup failed: {e}")

        # ------------------------------------------------------------------
        # Initialize YOLO-World ONNX Runtime session (GPU)
        # ------------------------------------------------------------------
        self.onnx_session = None
        self.onnx_input_name: str = "images"
        # ONNX output 구성을 동적으로 판별 (with / without NMS)
        self.onnx_output_names = ["num_dets", "labels", "scores", "boxes"]
        self.onnx_image_size = (640, 640)
        self.onnx_has_nms: bool = True

        try:
            onnx_path = onnx_model_path
            self.get_logger().info(f"[YOLO-ONNX] Initializing ONNX from: {onnx_path}")

            # Prefer pure ORT CUDA (TensorRT EP는 cuDNN 의존성 때문에 비활성화)
            available = ort.get_available_providers()
            providers: List[str] = []
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
                if p in available:
                    providers.append(p)

            self.get_logger().info(f"[YOLO-ONNX] Using ORT providers: {providers}")
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)

            # Infer input / output names from model
            try:
                inputs = self.onnx_session.get_inputs()
                if inputs:
                    self.onnx_input_name = inputs[0].name
                output_names = [o.name for o in self.onnx_session.get_outputs()]
                self.get_logger().info(f"[YOLO-ONNX] ONNX outputs: {output_names}")

                # Case 1: exported with NMS (num_dets, labels, scores, boxes)
                if all(n in output_names for n in ["num_dets", "labels", "scores", "boxes"]):
                    self.onnx_has_nms = True
                    self.onnx_output_names = ["num_dets", "labels", "scores", "boxes"]
                    self.get_logger().info("[YOLO-ONNX] Detected ONNX with built-in NMS.")
                # Case 2: exported without NMS (scores, boxes)
                elif all(n in output_names for n in ["scores", "boxes"]):
                    self.onnx_has_nms = False
                    self.onnx_output_names = ["scores", "boxes"]
                    self.get_logger().info("[YOLO-ONNX] Detected ONNX without NMS (scores, boxes).")
                else:
                    self.get_logger().warning(
                        f"[YOLO-ONNX] Unexpected ONNX outputs: {output_names} "
                        "(assuming no-NMS: scores, boxes)."
                    )
                    self.onnx_has_nms = False
                    # best-effort: try to use first 2 outputs
                    if len(output_names) >= 2:
                        self.onnx_output_names = output_names[:2]
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] Failed to inspect ONNX IO: {e}")

            self.get_logger().info("[YOLO-ONNX] ONNX Runtime session initialized.")
        except Exception as e:
            self.get_logger().error(f"[YOLO-ONNX] Failed to initialize ONNX session: {e}")
            self.onnx_session = None

        # ROS interfaces
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        self.viz_pub = self.create_publisher(Image, viz_topic, 10)
        self.det_pub = self.create_publisher(String, det_topic, 10)

        # Optional tracker
        try:
            self.tracker = sv.ByteTrack()
            self.get_logger().info("[YOLO-ONNX] ByteTrack initialized.")
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] ByteTrack disabled: {e}")
            self.tracker = None

    # ------------------------------------------------------------------
    # Image callback: run ONNX + SAM and publish
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image) -> None:
        t_start = time.perf_counter()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] Failed to convert image: {e}")
            return

        if self.onnx_session is None:
            return

        image = cv_image.copy()

        # --- YOLO-World ONNX inference ---
        t_yolo_start = time.perf_counter()
        try:
            dets = self._run_yolo_onnx(image)
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] ONNX inference failed: {e}")
            dets = []
        t_yolo_end = time.perf_counter()
        yolo_ms = (t_yolo_end - t_yolo_start) * 1000.0

        # --- Tracking (ByteTrack) ---
        tracker_ms = 0.0
        tracked_dets = dets
        if self.tracker is not None and len(dets) > 0:
            try:
                t_tracker_start = time.perf_counter()
                det_xyxy = np.asarray([d["bbox"] for d in dets], dtype=np.float32)
                det_cls = np.asarray([int(d.get("class_id", -1)) for d in dets], dtype=np.int32)
                det_conf = np.asarray([float(d.get("score", 0.0)) for d in dets], dtype=np.float32)
                detections_sv = sv.Detections(
                    xyxy=det_xyxy,
                    class_id=det_cls,
                    confidence=det_conf,
                )
                tracks = self.tracker.update_with_detections(detections_sv)
                tracked_dets = []
                for i in range(len(tracks)):
                    cid = int(tracks.class_id[i]) if tracks.class_id is not None else -1
                    score = float(tracks.confidence[i]) if tracks.confidence is not None else 0.0
                    tid = -1
                    if getattr(tracks, "tracker_id", None) is not None and i < len(tracks.tracker_id):
                        if tracks.tracker_id[i] is not None:
                            tid = int(tracks.tracker_id[i])
                    label = str(cid)
                    if 0 <= cid < len(self.yolo_texts):
                        label = self.yolo_texts[cid][0]
                    tracked_dets.append(
                        {
                            "label": label,
                            "score": score,
                            "bbox": tracks.xyxy[i].tolist(),
                            "class_id": cid,
                            "track_id": tid,
                        }
                    )
                t_tracker_end = time.perf_counter()
                tracker_ms = (t_tracker_end - t_tracker_start) * 1000.0
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] ByteTrack failed: {e}")

        # Run SAM for masks
        sam_masks_np = None
        sam_ms = 0.0
        if self.sam_model is not None and len(tracked_dets) > 0:
            try:
                bboxes = []
                for det in tracked_dets:
                    bbox = det.get("bbox")
                    if bbox is None or len(bbox) != 4:
                        continue
                    bboxes.append(bbox)

                if len(bboxes) > 0:
                    boxes_np = np.asarray(bboxes, dtype=np.float32)
                    try:
                        t_sam_start = time.perf_counter()
                        results = self.sam_model.predict(
                            image, bboxes=boxes_np, conf=self.sam_conf, verbose=False
                        )
                        t_sam_end = time.perf_counter()
                        sam_ms = (t_sam_end - t_sam_start) * 1000.0
                    except TypeError:
                        results = self.sam_model(image, bboxes=boxes_np, conf=self.sam_conf)

                    if results and getattr(results[0], "masks", None) is not None:
                        masks = results[0].masks
                        data = getattr(masks, "data", None)
                        if data is not None:
                            try:
                                sam_masks_np = data.cpu().numpy()
                            except Exception:
                                sam_masks_np = np.asarray(data)
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] SAM inference failed: {e}")

        # Draw masks (if any), then boxes and labels
        for idx, det in enumerate(tracked_dets):
            bbox = det.get("bbox")
            label = det.get("label", "")
            score = det.get("score", 0.0)
            track_id = det.get("track_id", -1)
            if bbox is None or len(bbox) != 4:
                continue

            if sam_masks_np is not None and idx < sam_masks_np.shape[0]:
                try:
                    mask = sam_masks_np[idx]
                    if mask is not None:
                        mask_bool = mask.astype(bool)
                        if mask_bool.shape[:2] == image.shape[:2]:
                            color = np.array([0, 0, 255], dtype=np.uint8)
                            alpha = 0.5
                            image[mask_bool] = (
                                (1.0 - alpha) * image[mask_bool] + alpha * color
                            ).astype(np.uint8)
                except Exception as e:
                    self.get_logger().warning(
                        f"[YOLO-ONNX] SAM mask overlay failed: {e}"
                    )

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if track_id is not None and track_id >= 0:
                text = f"{label} {score:.2f} ID:{track_id}"
            else:
                text = f"{label} {score:.2f}"
            cv2.putText(
                image,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(
                f"[YOLO-ONNX] Failed to convert annotated image: {e}"
            )
            return

        now = self.get_clock().now().to_msg()
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_color_optical_frame"
        self.viz_pub.publish(img_msg)

        # Attach segmentation payload directly into detections (single-section JSON).
        detections_payload = [dict(d) for d in tracked_dets]
        if sam_masks_np is not None:
            try:
                for idx, det in enumerate(detections_payload):
                    if idx >= sam_masks_np.shape[0]:
                        continue
                    mask = sam_masks_np[idx]
                    if mask is None:
                        continue

                    mask_u8 = (mask.astype(np.uint8) * 255)
                    ok, encoded = cv2.imencode(".png", mask_u8)
                    if not ok:
                        continue

                    det["mask_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
                    det["mask_height"] = int(mask_u8.shape[0])
                    det["mask_width"] = int(mask_u8.shape[1])
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] Failed to encode per-detection masks: {e}")

        # Single-section JSON publish: all info is inside detections[].
        try:
            det_msg = String()
            det_msg.data = json.dumps(
                {
                    # Preserve camera source header stamp as-is.
                    "header": {
                        "stamp": {
                            "sec": int(msg.header.stamp.sec),
                            "nanosec": int(msg.header.stamp.nanosec),
                        },
                        "frame_id": msg.header.frame_id,
                    },
                    "detections": detections_payload,
                },
                ensure_ascii=False,
            )
            self.det_pub.publish(det_msg)
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] Failed to publish detections: {e}")

        # --- Timing log ---
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        self.get_logger().info(
            f"[YOLO-ONNX] Timing: yolo={yolo_ms:.1f}ms, tracker={tracker_ms:.1f}ms, "
            f"sam={sam_ms:.1f}ms, total={total_ms:.1f}ms"
        )

    # ------------------------------------------------------------------
    # YOLO-World ONNX helpers
    # ------------------------------------------------------------------
    def _preprocess_for_onnx(self, image_bgr: np.ndarray):
        size = self.onnx_image_size
        h, w = image_bgr.shape[:2]
        max_size = max(h, w)
        scale_factor = size[0] / max_size
        pad_h = (max_size - h) // 2
        pad_w = (max_size - w) // 2

        pad_image = np.zeros((max_size, max_size, 3), dtype=image_bgr.dtype)
        # BGR -> RGB
        pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image_bgr[:, :, [2, 1, 0]]

        image = cv2.resize(
            pad_image,
            size,
            interpolation=cv2.INTER_LINEAR,
        ).astype("float32")
        image /= 255.0
        image = image[None]  # (1, H, W, 3)
        return image, scale_factor, (pad_h, pad_w)

    def _run_yolo_onnx(self, image_bgr: np.ndarray):
        if self.onnx_session is None:
            return []

        h, w = image_bgr.shape[:2]
        image, scale_factor, pad_param = self._preprocess_for_onnx(image_bgr)

        input_ort = ort.OrtValue.ortvalue_from_numpy(
            image.transpose((0, 3, 1, 2))
        )

        if self.onnx_has_nms:
            # With NMS: [num_dets, labels, scores, boxes]
            num_dets, labels, scores, bboxes = self.onnx_session.run(
                self.onnx_output_names, {self.onnx_input_name: input_ort}
            )

            num_dets = int(num_dets[0][0])
            labels = labels[0, :num_dets]
            scores = scores[0, :num_dets]
            bboxes = bboxes[0, :num_dets]
        else:
            # Without NMS: [scores, boxes] (YOLO-World ONNX demo 스타일)
            scores_raw, bboxes_raw = self.onnx_session.run(
                self.onnx_output_names, {self.onnx_input_name: input_ort}
            )
            # scores_raw: (1, N, C), bboxes_raw: (1, N, 4)
            scores_t = torch.from_numpy(scores_raw[0])
            boxes_t = torch.from_numpy(bboxes_raw[0])

            device_torch = "cuda" if "cuda" in str(self.cfg.device) else "cpu"
            scores_t = scores_t.to(device_torch)
            boxes_t = boxes_t.to(device_torch)

            # 1) 각 박스별 최고 점수/클래스
            max_scores, labels_max = torch.max(scores_t, dim=1)
            keep_mask = max_scores > self.yolo_score_thr

            if not keep_mask.any():
                return []

            filtered_boxes = boxes_t[keep_mask]
            filtered_scores = max_scores[keep_mask]
            filtered_labels = labels_max[keep_mask]

            # 1.5) Pre-NMS Top-K 필터링으로 NMS 연산량 감소
            if len(filtered_scores) > self.yolo_topk * 2:
                _, topk_idx = torch.topk(filtered_scores, k=self.yolo_topk * 2)
                filtered_boxes = filtered_boxes[topk_idx]
                filtered_scores = filtered_scores[topk_idx]
                filtered_labels = filtered_labels[topk_idx]

            # 2) batched_nms 사용 (있으면), 없으면 단일 NMS fallback
            nms_thr = 0.7
            if batched_nms is not None:
                keep_idx = batched_nms(
                    filtered_boxes, filtered_scores, filtered_labels, iou_threshold=nms_thr
                )
            elif nms is not None:
                keep_idx = nms(filtered_boxes, filtered_scores, iou_threshold=nms_thr)
            else:
                # torchvision.ops 없음 → simple score top-k만 사용
                _, keep_idx = torch.topk(filtered_scores, k=min(self.yolo_topk, len(filtered_scores)))

            boxes_np = filtered_boxes[keep_idx].cpu().numpy()
            scores_np = filtered_scores[keep_idx].cpu().numpy()
            labels_np = filtered_labels[keep_idx].cpu().numpy()

            num_dets = int(scores_np.shape[0])
            bboxes = boxes_np
            scores = scores_np.astype(np.float32)
            labels = labels_np.astype(np.int64)

        # Undo padding / scaling
        if num_dets > 0:
            bboxes = bboxes.astype(float)
            bboxes -= np.array(
                [pad_param[1], pad_param[0], pad_param[1], pad_param[0]]
            )
            bboxes /= scale_factor
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
            bboxes = bboxes.round().astype("int")

        dets = []
        for i in range(num_dets):
            x1, y1, x2, y2 = bboxes[i].tolist()
            class_id = int(labels[i])
            score = float(scores[i])

            name = str(class_id)
            try:
                if 0 <= class_id < len(self.yolo_texts):
                    name = self.yolo_texts[class_id][0]
            except Exception:
                pass

            dets.append(
                {
                    "label": name,
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                }
            )

        return dets


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloWorldOnnxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()