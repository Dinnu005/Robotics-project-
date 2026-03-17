# Setup Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| pip | ≥ 23.0 |
| Webcam (optional) | any USB / built-in |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Dinnu005/Robotics-project-.git
cd Robotics-project-
```

---

## 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` pulls in PyTorch (~200 MB on first install).
> If you are on a slow connection, install only the OpenCV stack first:
> ```bash
> pip install opencv-python numpy
> ```

---

## 4. Run the Demo (No Camera Required)

```bash
python demo/run_demo.py
```

A menu appears:

```
══════════════════════════════════════════════════
  Autonomous Weed Removal – Vision Demo
  Project: Rail-based Robot Under Solar Panels
══════════════════════════════════════════════════

  [1]  OpenCV HSV Detection  (fast, no internet)
  [2]  YOLOv8 Detection      (accurate, ~6 MB download)
  [Q]  Quit

  Select mode →
```

Select **1** for an immediate synthetic demo.

---

## 5. Test on Your Own Video / Images

```bash
# OpenCV – video file
python vision/basic_detection.py --source path/to/video.mp4 --show-mask

# OpenCV – webcam (index 0)
python vision/basic_detection.py --source 0

# YOLO – image
python vision/yolo_detection.py --source path/to/image.jpg
```

---

## 6. Prepare a Dataset and Train YOLO

See **[docs/dataset_guide.md](dataset_guide.md)** for full instructions.

Quick summary:

```bash
# Download a weed dataset from Roboflow
python dataset/download_dataset.py roboflow --api-key YOUR_KEY

# Train YOLOv8 nano (fast, good for laptop demo)
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

## 7. Connect to Arduino (Optional)

1. Upload `arduino/robot_control.ino` to your Arduino Uno/Mega using the
   Arduino IDE.
2. Find your serial port:
   - Windows: check Device Manager → COM ports (e.g. `COM3`)
   - Linux/macOS: `ls /dev/tty*` (e.g. `/dev/ttyUSB0`)
3. Pass the port to the vision script:

```bash
python vision/basic_detection.py --source 0
# Then in robot_controller.py, set serial_port='COM3'
```

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `cv2.error: ... Can't open camera` | Change `--source 0` to `--source 1` or `--source demo` |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| Window doesn't appear | Install a display backend: `pip install opencv-python` (not headless) |
| YOLO download fails | Check internet connection; weights are ~6 MB from GitHub |
| Arduino not detected | Check USB cable and drivers; try a different COM port |
