<div align="center">

# 🤖 Autonomous Weed Removal Robot
### Beneath Solar Panel Structures

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-orange)](https://ultralytics.com)
[![Arduino](https://img.shields.io/badge/Arduino-Control-teal?logo=arduino)](https://arduino.cc)

**A fully functional vision + control system for a rail-based weed removal robot.**  
*Runs entirely on a laptop — no hardware required for the demo.*

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Repository Structure](#-repository-structure)
3. [Quick Start — Run the Demo](#-quick-start--run-the-demo-in-3-steps)
4. [Vision System Architecture](#-vision-system-architecture)
   - [Stage 1: OpenCV HSV Detection](#stage-1-opencv-hsv-detection)
   - [Stage 2: YOLOv8 Deep Learning](#stage-2-yolov8-deep-learning-detection)
5. [Dataset Preparation](#-dataset-preparation)
6. [Training Your Own YOLO Model](#-training-your-own-yolo-model)
7. [Robot Control System](#-robot-control-system)
8. [Arduino Integration](#-arduino-integration)
9. [Project Status](#-project-status)
10. [References](#-references)

---

## 🌿 Project Overview

Solar farms require regular weed clearance beneath panel arrays, but the
**low clearance** (< 60 cm), **uneven terrain**, and **risk of panel damage**
make manual weeding dangerous and costly.

This project implements an **autonomous rail-guided robot** that:

1. **Sees** — computer vision detects weeds in real time
2. **Decides** — control logic determines MOVE / STOP / CUT actions
3. **Acts** — Arduino drives the BLDC cutting blade and locomotion motors

```
Camera ──► Vision (OpenCV / YOLOv8) ──► Robot Controller ──► Arduino ──► Motors
```

### Key Hardware (Physical Prototype)

| Component | Specification |
|-----------|--------------|
| Locomotion | Rail-guided drive system |
| Cutting Blade | TCT Saw Blade — Ø 255 mm, 80 teeth |
| Drive Motor | BLDC CY-1518 — 48 V, 1500 W, 3000–5000 RPM |
| Controller | Arduino Uno / Mega |
| Camera | USB webcam or Raspberry Pi Camera |

---

## 📁 Repository Structure

```
Robotics-project-/
│
├── vision/                        ← Computer vision modules
│   ├── __init__.py
│   ├── basic_detection.py         ← Stage 1: OpenCV HSV weed detection
│   ├── yolo_detection.py          ← Stage 2: YOLOv8 deep learning detection
│   └── robot_controller.py        ← Simulated & real robot command dispatcher
│
├── dataset/                       ← Dataset tools
│   └── download_dataset.py        ← Download & organise training data
│
├── demo/                          ← Demo launcher
│   └── run_demo.py                ← Interactive demo menu (no hardware needed)
│
├── arduino/                       ← Embedded control
│   └── robot_control.ino          ← Arduino sketch (serial command protocol)
│
├── docs/                          ← Documentation
│   ├── setup_guide.md             ← Step-by-step installation guide
│   └── dataset_guide.md           ← Dataset sources + labelling instructions
│
├── requirements.txt               ← Python dependencies
├── PPT_text                       ← Original project presentation text
└── README.md                      ← This file
```

---

## 🚀 Quick Start — Run the Demo in 3 Steps

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> Minimum install (no YOLO, just OpenCV):
> ```bash
> pip install opencv-python numpy
> ```

### Step 2 — Run the demo

```bash
python demo/run_demo.py
```

This opens an **interactive menu** — select option **1** (OpenCV) for an
**instant demo with a synthetic weed scene** (no camera needed).

### Step 3 — See the results

The window shows:
- 🟩 **Green bounding boxes** around detected weeds
- 📊 **Coverage percentage** (how much of the frame is weed)
- 🤖 **Robot action** (MOVE / STOP / CUT) displayed on-screen
- Terminal prints a timestamped action log

```
  [14:23:05]  🟢 MOVE  – robot moving forward
  [14:23:07]  🔴 STOP  – weed detected, robot stopped
  [14:23:08]  ⚡ CUT   – cutting blade activated
```

---

## 🔬 Vision System Architecture

The vision pipeline has **two stages** — start with Stage 1 (works immediately,
no internet), then upgrade to Stage 2 (higher accuracy after training).

### Stage 1: OpenCV HSV Detection

**File:** `vision/basic_detection.py`

#### How It Works (step by step)

```
BGR Frame
    │
    ▼
Convert to HSV          ← separates colour (hue) from brightness
    │
    ▼
Green Colour Mask       ← Hue: 35–85 | Sat: 40–255 | Val: 40–255
    │
    ▼
Morphological Cleanup   ← remove noise pixels, fill small gaps
    │
    ▼
Find Contours           ← connected green blob boundaries
    │
    ▼
Filter by Area          ← ignore blobs < 500 px² (dust, reflections)
    │
    ▼
Compute Coverage %      ← (green pixels / total pixels) × 100
    │
    ▼
Decide Action           ← < 5% → MOVE | 5–15% → STOP | ≥ 15% → CUT
    │
    ▼
Draw Bounding Boxes     ← annotated output frame
```

#### Run It

```bash
# Synthetic demo (no camera)
python vision/basic_detection.py --source demo --show-mask

# Webcam
python vision/basic_detection.py --source 0

# Video file
python vision/basic_detection.py --source path/to/video.mp4

# Single image
python vision/basic_detection.py --source path/to/image.jpg
```

#### Tuning the HSV Range

If detection is inaccurate on your footage, adjust the green range in
`vision/basic_detection.py`:

```python
HSV_LOWER_GREEN = np.array([35, 40, 40])   # lower bound (H, S, V)
HSV_UPPER_GREEN = np.array([85, 255, 255]) # upper bound (H, S, V)
```

| Parameter | Effect |
|-----------|--------|
| Lower Hue (35) | Raise to exclude yellow-green |
| Upper Hue (85) | Lower to exclude teal/blue-green |
| Lower Saturation (40) | Raise to ignore pale colours |
| Lower Value (40) | Raise to ignore dark shadows |

---

### Stage 2: YOLOv8 Deep Learning Detection

**File:** `vision/yolo_detection.py`

YOLOv8 (You Only Look Once, version 8) is a **real-time object detector**
that uses a convolutional neural network to find objects directly.

#### How YOLO Works

```
Input Frame (640×640)
        │
        ▼
Backbone (CSPDarknet)   ← extracts visual features at multiple scales
        │
        ▼
Feature Pyramid Network ← combines small + large scale features
        │
        ▼
Detection Head          ← predicts (x, y, w, h, confidence, class)
        │
        ▼
Non-Max Suppression     ← remove overlapping duplicate boxes
        │
        ▼
Bounding Boxes + Labels ← final detections on original frame
```

#### Why YOLO is Better than HSV

| | OpenCV HSV | YOLOv8 |
|--|-----------|--------|
| Accuracy | Medium | High |
| Lighting sensitivity | High | Low |
| Shadow handling | Poor | Good |
| Training required | No | Yes (~500 images) |
| Speed | Very fast | Fast (25–60 FPS) |

#### Run It

```bash
# Pretrained weights (detects general objects)
python vision/yolo_detection.py --source demo

# Custom trained weights (weed-specific)
python vision/yolo_detection.py \
  --weights runs/detect/weed_detector/weights/best.pt \
  --source 0
```

---

## 📦 Dataset Preparation

To train YOLO specifically on weed images, you need a labelled dataset.

### Recommended Free Datasets

| Dataset | Images | Best For |
|---------|--------|----------|
| [Roboflow Weed Detection](https://universe.roboflow.com/roboflow-100/weeds-nfvsp) | ~1 400 | Quick start |
| [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) | 17 509 | Variety |
| [Plant Seedlings (Kaggle)](https://www.kaggle.com/c/plant-seedlings-classification) | 5 539 | Dense growth |

### Download Script

```bash
# Option A – Roboflow (get a free API key at roboflow.com)
python dataset/download_dataset.py roboflow --api-key YOUR_KEY

# Option B – DeepWeeds (no key, ~1.5 GB)
python dataset/download_dataset.py deepweeds

# Option C – Your own images
python dataset/download_dataset.py local --raw-dir dataset/raw_images
```

The script automatically organises images into:
```
dataset/
├── images/ train/ val/ test/
├── labels/ train/ val/ test/
└── data.yaml
```

See **[docs/dataset_guide.md](docs/dataset_guide.md)** for full details
including manual labelling with LabelImg.

---

## 🏋️ Training Your Own YOLO Model

After downloading the dataset:

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=weed_detector
```

Expected results after 50 epochs:

```
Epoch 50/50:  mAP50=0.82  precision=0.85  recall=0.79
Results saved to runs/detect/weed_detector/
```

Use your model:

```bash
python vision/yolo_detection.py \
  --weights runs/detect/weed_detector/weights/best.pt \
  --source 0
```

---

## 🎮 Robot Control System

**File:** `vision/robot_controller.py`

The `RobotController` class translates vision output into robot commands:

| Vision Input | Action | Robot Behaviour |
|-------------|--------|----------------|
| Coverage < 5% | `MOVE` | Drive motors ON, blade OFF |
| Coverage 5–15% | `STOP` | All motors OFF |
| Coverage ≥ 15% | `CUT` | Blade ON, drive motors OFF |
| YOLO high confidence weed | `CUT` | Blade ON |

All commands are timestamped and saved to `robot_log.txt`.

---

## 🔌 Arduino Integration

When you have the physical hardware:

1. Upload `arduino/robot_control.ino` to your Arduino
2. Connect laptop ↔ Arduino via USB
3. Enable serial in the controller:

```python
from vision.robot_controller import RobotController

# Windows: 'COM3'   |   Linux: '/dev/ttyUSB0'   |   macOS: '/dev/tty.usbmodem...'
controller = RobotController(serial_port='COM3', baud_rate=9600)
```

### Serial Protocol

```
Python →  'M'  → Arduino → Drive motors ON, blade OFF
Python →  'S'  → Arduino → All motors OFF
Python →  'C'  → Arduino → Blade ON, drive motors OFF
```

### Wiring Diagram

```
Laptop (USB)
    │
    └──► Arduino Uno/Mega
              │
              ├─ Pin 2 ──► L298N ENA (drive motor enable)
              ├─ Pin 3 ──► L298N IN1 (direction)
              ├─ Pin 4 ──► L298N IN2 (direction)
              └─ Pin 5 ──► BLDC Driver Enable
                                │
                                └──► BLDC Motor (48V, external supply)
```

---

## 📊 Project Status

| Component | Status |
|-----------|--------|
| ✅ Concept design (CAD) | Complete |
| ✅ Component selection | Complete |
| ✅ OpenCV weed detection | **Complete** |
| ✅ YOLO detection module | **Complete** |
| ✅ Dataset download script | **Complete** |
| ✅ Robot action simulation | **Complete** |
| ✅ Arduino sketch | **Complete** |
| ⏳ YOLO model fine-tuning | In progress |
| ⏳ Physical fabrication | Pending |
| ⏳ Real-world testing | Pending |

---

## 📚 References

1. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
2. OpenCV Documentation — https://docs.opencv.org
3. DeepWeeds Dataset — A. Olsen et al., *Scientific Reports* (2019)
4. Roboflow Universe — https://universe.roboflow.com
5. Arduino Serial Communication — https://www.arduino.cc/reference/en/language/functions/communication/serial/
