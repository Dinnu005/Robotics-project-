"""
basic_detection.py
==================
Stage 1 – OpenCV HSV-based weed detection.

What this script does
---------------------
1. Reads a live webcam feed (or a video file / single image).
2. Converts every frame from BGR to HSV colour space.
3. Applies a green-colour mask to isolate vegetation.
4. Finds contours of the masked regions.
5. Draws labelled bounding boxes around detected weeds.
6. Computes a *weed coverage percentage* for the frame.
7. Triggers a simulated robot action (MOVE / STOP / CUT) based on that
   percentage and sends the command string to the RobotController.

Run directly
------------
    python vision/basic_detection.py                    # webcam
    python vision/basic_detection.py --source video.mp4
    python vision/basic_detection.py --source image.jpg
    python vision/basic_detection.py --source demo      # synthetic demo frame

Press  Q  to quit the live window.
"""

import argparse
import sys
import cv2
import numpy as np

from .robot_controller import RobotController

# ── HSV colour range for green vegetation ────────────────────────────────────
# Hue: 35-85  covers yellow-green → deep green
# Sat: 40-255 filters out white/grey sky reflections
# Val: 40-255 filters out very dark (shadow) pixels
HSV_LOWER_GREEN = np.array([35, 40, 40], dtype=np.uint8)
HSV_UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

# Morphological kernel – removes tiny noise blobs
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Detection thresholds (weed coverage % of the frame area)
THRESHOLD_STOP = 5.0   # >= 5 % → stop the robot
THRESHOLD_CUT = 15.0   # >= 15 % → activate cutting blade


class WeedDetector:
    """
    OpenCV-based green-vegetation detector.

    Parameters
    ----------
    min_contour_area : int
        Ignore contours smaller than this (filters out noise).
    show_mask : bool
        If True, show the binary HSV mask in a separate window.
    """

    def __init__(self, min_contour_area: int = 500, show_mask: bool = False):
        self.min_contour_area = min_contour_area
        self.show_mask = show_mask
        self.controller = RobotController()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _build_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a cleaned binary mask for green regions in *frame*."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
        # Remove noise, then fill small holes
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, MORPH_KERNEL)
        return cleaned

    def _find_weed_contours(self, mask: np.ndarray):
        """Return contours that exceed the minimum area threshold."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

    def _coverage_percent(self, mask: np.ndarray) -> float:
        """Return the percentage of the frame covered by green pixels."""
        total_pixels = mask.shape[0] * mask.shape[1]
        green_pixels = int(np.count_nonzero(mask))
        return round(100.0 * green_pixels / total_pixels, 2)

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Analyse a single BGR frame.

        Returns
        -------
        dict with keys:
            annotated  – frame with bounding boxes drawn
            mask       – binary green mask
            contours   – list of OpenCV contours
            coverage   – weed coverage % (float)
            action     – robot command string
        """
        mask = self._build_mask(frame)
        contours = self._find_weed_contours(mask)
        coverage = self._coverage_percent(mask)

        # Determine robot action
        if coverage >= THRESHOLD_CUT:
            action = "CUT"
        elif coverage >= THRESHOLD_STOP:
            action = "STOP"
        else:
            action = "MOVE"

        self.controller.execute(action)

        # Draw annotations on a copy of the frame
        annotated = frame.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"WEED ({cv2.contourArea(cnt):.0f}px²)",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # HUD overlay
        hud_color = (0, 0, 255) if action in ("STOP", "CUT") else (0, 220, 0)
        cv2.putText(
            annotated,
            f"Coverage: {coverage:.1f}%   Action: {action}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            hud_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Weeds found: {len(contours)}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return {
            "annotated": annotated,
            "mask": mask,
            "contours": contours,
            "coverage": coverage,
            "action": action,
        }

    def run(self, source=0):
        """
        Run the detector on a webcam index, video path, or image path.
        Pass  source='demo'  to synthesise a test frame without a camera.
        """
        # ── synthetic demo mode ──────────────────────────────────────────────
        if source == "demo":
            frame = _make_demo_frame()
            result = self.process_frame(frame)
            _show_result(result, self.show_mask)
            print(f"\n[Demo] Coverage: {result['coverage']}%  →  Action: {result['action']}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # ── image mode ───────────────────────────────────────────────────────
        if isinstance(source, str):
            img = cv2.imread(source)
            if img is None:
                sys.exit(f"[ERROR] Cannot load image: {source}")
            result = self.process_frame(img)
            _show_result(result, self.show_mask)
            print(f"[Image] Coverage: {result['coverage']}%  →  Action: {result['action']}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        # ── video / webcam mode ──────────────────────────────────────────────
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Cannot open video source: {source}")

        print("[INFO] Running weed detector … press Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            cv2.imshow("Weed Detection – OpenCV", result["annotated"])
            if self.show_mask:
                cv2.imshow("Green Mask", result["mask"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ── helpers ────────────────────────────────────────────────────────────────

def _make_demo_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic BGR frame with green patches to simulate weeds.
    Useful for quick testing without a camera or video file.
    """
    # Brown soil background
    frame = np.full((height, width, 3), (34, 85, 85), dtype=np.uint8)

    # Solar panel shadow bands (dark grey)
    for y_start in range(0, height, 90):
        frame[y_start:y_start + 20, :] = (40, 40, 40)

    # Green weed patches
    patches = [
        (80, 120, 60, 50),
        (250, 200, 80, 60),
        (420, 100, 50, 45),
        (150, 300, 100, 70),
        (500, 350, 70, 55),
    ]
    for (px, py, pw, ph) in patches:
        # Random-ish green hue in HSV → convert to BGR for drawing
        hsv_patch = np.full((ph, pw, 3), (55, 180, 160), dtype=np.uint8)
        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        # Add slight variation
        noise = np.random.randint(-20, 20, bgr_patch.shape, dtype=np.int16)
        bgr_patch = np.clip(bgr_patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame[py:py + ph, px:px + pw] = bgr_patch

    return frame


def _show_result(result: dict, show_mask: bool):
    cv2.imshow("Weed Detection – OpenCV", result["annotated"])
    if show_mask:
        cv2.imshow("Green Mask", result["mask"])


# ── CLI entry point ────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="OpenCV weed detector")
    parser.add_argument(
        "--source",
        default=0,
        help="Video source: webcam index (0), video path, image path, or 'demo'",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area to consider a weed (px²)",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the green HSV mask in a separate window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = args.source
    # Convert numeric string to int for webcam index
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    detector = WeedDetector(min_contour_area=args.min_area, show_mask=args.show_mask)
    detector.run(source=source)
