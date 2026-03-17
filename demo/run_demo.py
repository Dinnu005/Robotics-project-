"""
run_demo.py
===========
Quick demo launcher for the weed-detection system.

What this script does
---------------------
Presents a menu to choose between:
  1. OpenCV HSV detection  (no GPU, no internet required)
  2. YOLOv8 detection      (auto-downloads ~6 MB weights on first run)

Both modes synthesise a test frame when no camera or video is available,
so you can demonstrate the system to your faculty immediately on a laptop
with no external hardware.

Usage
-----
  python demo/run_demo.py              # interactive menu
  python demo/run_demo.py --mode opencv
  python demo/run_demo.py --mode yolo
  python demo/run_demo.py --mode opencv --source 0        # webcam
  python demo/run_demo.py --mode opencv --source video.mp4
"""

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on sys.path so that `vision` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_opencv(source):
    from vision.basic_detection import WeedDetector  # noqa: PLC0415
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Stage 1 – OpenCV HSV Weed Detection")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print("  How it works:")
    print("  1. Convert frame from BGR → HSV colour space")
    print("  2. Apply green-colour mask (Hue 35–85, S>40, V>40)")
    print("  3. Find contours of green blobs")
    print("  4. Draw bounding boxes & compute coverage %")
    print("  5. Decide MOVE / STOP / CUT based on coverage\n")
    detector = WeedDetector(show_mask=True)
    detector.run(source=source)


def _run_yolo(source):
    from vision.yolo_detection import YOLOWeedDetector  # noqa: PLC0415
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Stage 2 – YOLOv8 Deep Learning Detection")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print("  How it works:")
    print("  1. Run YOLOv8 neural network on each frame")
    print("  2. Network predicts class + bounding box + confidence")
    print("  3. Filter detections by confidence threshold (30 %)")
    print("  4. Decide MOVE / STOP / CUT based on detections\n")
    print("  Note: First run auto-downloads yolov8n.pt (~6 MB)\n")
    detector = YOLOWeedDetector()
    detector.run(source=source)


def _interactive_menu(source):
    print("\n" + "═" * 50)
    print("  Autonomous Weed Removal – Vision Demo")
    print("  Project: Rail-based Robot Under Solar Panels")
    print("═" * 50)
    print()
    print("  [1]  OpenCV HSV Detection  (fast, no internet)")
    print("  [2]  YOLOv8 Detection      (accurate, ~6 MB download)")
    print("  [Q]  Quit")
    print()
    choice = input("  Select mode → ").strip().upper()

    if choice == "1":
        _run_opencv(source)
    elif choice == "2":
        _run_yolo(source)
    elif choice == "Q":
        print("Bye!")
    else:
        print("Invalid choice.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Weed detection demo")
    parser.add_argument(
        "--mode",
        choices=["opencv", "yolo"],
        default=None,
        help="Detection mode (omit for interactive menu)",
    )
    parser.add_argument(
        "--source",
        default="demo",
        help="Source: 'demo' (default), webcam index (0), video path, or image path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if args.mode == "opencv":
        _run_opencv(source)
    elif args.mode == "yolo":
        _run_yolo(source)
    else:
        _interactive_menu(source)
