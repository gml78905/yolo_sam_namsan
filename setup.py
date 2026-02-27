from setuptools import setup, find_packages
import glob

package_name = "yolo_sam"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(include=["yolo_sam", "yolo_world_test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob.glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="urorin",
    maintainer_email="todo@example.com",
    description=(
        "YOLO-World test node that subscribes to /go1_d435/color/image_raw "
        "and publishes visualization images to visualization/test."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_sam_node = yolo_sam.yolo_sam_node:main",
        ],
    },
)

