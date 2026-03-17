"""
yolo_detection.py
=================
Stage 2 – YOLOv8 deep-learning weed detection.

What this script does
---------------------
1. Loads a YOLOv8 model (nano by default; swap to a custom-trained model
   once you have one – see docs/dataset_guide.md).
2. Runs inference on each frame from a webcam, video, or image.
3. Draws labelled bounding boxes with confidence scores for every
   detected weed.
4. Passes the highest-confidence detection's class to the RobotController
   to trigger MOVE / STOP / CUT.

Installation
------------
    pip install ultralytics opencv-python

The first time you run this script, Ultralytics will auto-download the
YOLOv8n weights (~6 MB) from GitHub.

To use your own trained model:
    python vision/yolo_detection.py --weights runs/train/exp/weights/best.pt

Run directly
------------
    python vision/yolo_detection.py                    # webcam
    python vision/yolo_detection.py --source video.mp4
    python vision/yolo_detection.py --source image.jpg
    python vision/yolo_detection.py --source demo      # synthetic frame

Press  Q  to quit the live window.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .robot_controller import RobotController

# ── constants ──────────────────────────────────────────────────────────────

# Default pretrained weights – replaced automatically with a fine-tuned
# weed-detection model once you complete the training pipeline.
DEFAULT_WEIGHTS = "yolov8n.pt"

# Minimum confidence to draw a box
CONF_THRESHOLD = 0.30

# Colour palette for bounding boxes (one per class index)
BOX_COLOURS = [
    (0, 255, 0),    # class 0 – weed (green)
    (0, 165, 255),  # class 1 – soil (orange)
    (255, 0, 0),    # class 2 – other (blue)
]


class YOLOWeedDetector:
    """
    YOLOv8-based weed detector.

    Parameters
    ----------
    weights : str | Path
        Path to a YOLOv8 .pt weights file or a model name like 'yolov8n.pt'.
    conf : float
        Confidence threshold (0–1).
    device : str
        Inference device – 'cpu', '0' (GPU 0), etc.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        conf: float = CONF_THRESHOLD,
        device: str = "cpu",
    ):
        self.conf = conf
        self.device = device
        self.controller = RobotController()
        self.model = self._load_model(weights)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _load_model(self, weights: str):
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError:
            sys.exit(
                "[ERROR] ultralytics not installed.\n"
                "        Run:  pip install ultralytics"
            )
        print(f"[INFO] Loading YOLO model: {weights}")
        model = YOLO(weights)
        model.to(self.device)
        return model

    def _get_box_colour(self, class_id: int) -> tuple:
        return BOX_COLOURS[class_id % len(BOX_COLOURS)]

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run YOLO inference on a single BGR frame.

        Returns
        -------
        dict with keys:
            annotated   – frame with bounding boxes drawn
            detections  – list of dicts {label, conf, bbox}
            action      – robot command string
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated = frame.copy()
        detections = []
        top_conf = 0.0
        top_label = ""

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = (
                    self.model.names[cls_id]
                    if hasattr(self.model, "names") and self.model.names
                    else f"cls{cls_id}"
                )
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})

                colour = self._get_box_colour(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                tag = f"{label} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(
                    annotated,
                    tag,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                if conf > top_conf:
                    top_conf = conf
                    top_label = label

        # Decide robot action based on whether high-confidence weed is detected
        weed_labels = {"weed", "vegetation", "plant"}
        is_weed = any(
            d["label"].lower() in weed_labels and d["conf"] >= 0.5
            for d in detections
        )
        if is_weed and top_conf >= 0.7:
            action = "CUT"
        elif detections:
            action = "STOP"
        else:
            action = "MOVE"

        self.controller.execute(action)

        # HUD
        hud_col = (0, 0, 255) if action in ("STOP", "CUT") else (0, 220, 0)
        cv2.putText(
            annotated,
            f"Detections: {len(detections)}   Action: {action}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            hud_col,
            2,
            cv2.LINE_AA,
        )

        return {"annotated": annotated, "detections": detections, "action": action}

    def run(self, source=0):
        """
        Run the YOLO detector on a webcam index, video path, or image path.
        Pass  source='demo'  to use a synthetic test frame.
        """
        # ── synthetic demo mode ──────────────────────────────────────────────
        if source == "demo":
            frame = _make_demo_frame()
            result = self.process_frame(frame)
            cv2.imshow("Weed Detection – YOLOv8", result["annotated"])
            print(f"\n[Demo] Detections: {len(result['detections'])}  →  Action: {result['action']}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # ── image mode ───────────────────────────────────────────────────────
        if isinstance(source, str) and Path(source).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
        }:
            img = cv2.imread(source)
            if img is None:
                sys.exit(f"[ERROR] Cannot load image: {source}")
            result = self.process_frame(img)
            cv2.imshow("Weed Detection – YOLOv8", result["annotated"])
            print(f"[Image] Detections: {len(result['detections'])}  →  Action: {result['action']}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # ── video / webcam mode ──────────────────────────────────────────────
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open video source: {source}")

        print("[INFO] Running YOLO weed detector … press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            cv2.imshow("Weed Detection – YOLOv8", result["annotated"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ── helpers ────────────────────────────────────────────────────────────────

def _make_demo_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Synthetic soil-with-green-patches frame for offline testing."""
    frame = np.full((height, width, 3), (34, 85, 85), dtype=np.uint8)
    patches = [(80, 120, 60, 50), (250, 200, 80, 60), (420, 100, 50, 45)]
    for px, py, pw, ph in patches:
        hsv_patch = np.full((ph, pw, 3), (55, 180, 160), dtype=np.uint8)
        frame[py:py + ph, px:px + pw] = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
    return frame


# ── CLI entry point ────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 weed detector")
    parser.add_argument("--source", default=0, help="Webcam index, video/image path, or 'demo'")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pt weights file")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu or 0 (GPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    detector = YOLOWeedDetector(weights=args.weights, conf=args.conf, device=args.device)
    detector.run(source=source)
