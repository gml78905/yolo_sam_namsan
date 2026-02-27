#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    visualization_topic = LaunchConfiguration("visualization_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("yolo_sam"), "config", "yolo_sam.yaml"]
                ),
                description="Path to yolo_sam ROS parameters YAML",
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/go1_d435/color/image_raw",
                description="Input image topic",
            ),
            DeclareLaunchArgument(
                "visualization_topic",
                default_value="/visualization/test",
                description="Annotated image output topic",
            ),
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/yolo_sam/detections",
                description="Detections JSON output topic",
            ),
            Node(
                package="yolo_sam",
                executable="yolo_sam_node",
                name="yolo_sam_node",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "image_topic": image_topic,
                        "visualization_topic": visualization_topic,
                        "detections_topic": detections_topic,
                    }
                ],
            ),
        ]
    )
