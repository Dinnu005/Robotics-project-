# Vision module for autonomous weed detection
# Supports OpenCV HSV-based detection and YOLOv8 deep learning detection

from .basic_detection import WeedDetector
from .yolo_detection import YOLOWeedDetector
from .robot_controller import RobotController

__all__ = ["WeedDetector", "YOLOWeedDetector", "RobotController"]
