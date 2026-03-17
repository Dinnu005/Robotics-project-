# Dataset Guide – Weed Detection for YOLO Training

This document explains where to get labelled weed images and how to
prepare them for YOLOv8 fine-tuning.

---

## 1. Recommended Public Datasets

| Dataset | Images | Classes | License | Link |
|---------|--------|---------|---------|------|
| **Crop and Weed Detection (Roboflow)** | ~1 400 | 2 (crop, weed) | CC BY 4.0 | [Link](https://universe.roboflow.com/roboflow-100/weeds-nfvsp) |
| **DeepWeeds** | 17 509 | 9 weed species | CC BY 4.0 | [GitHub](https://github.com/AlexOlsen/DeepWeeds) |
| **Plant Seedlings Dataset (Kaggle)** | 5 539 | 12 plant species | CC0 | [Kaggle](https://www.kaggle.com/c/plant-seedlings-classification) |
| **Weed Detection in Soybean (Roboflow)** | ~3 000 | 3 (weed, crop, soil) | CC BY 4.0 | [Link](https://universe.roboflow.com/brad-dwyer/weed-detection-in-soybean) |

---

## 2. Downloading via the Script

```bash
# Option A – Roboflow (recommended for solar-panel weed context)
python dataset/download_dataset.py roboflow --api-key YOUR_KEY

# Option B – DeepWeeds (no API key, ~1.5 GB)
python dataset/download_dataset.py deepweeds

# Option C – Your own images
python dataset/download_dataset.py local --raw-dir dataset/raw_images
```

After download the directory looks like:

```
dataset/
├── images/
│   ├── train/   ← 80 % of images
│   ├── val/     ← 10 %
│   └── test/    ← 10 %
├── labels/
│   ├── train/   ← YOLO .txt label files (one per image)
│   ├── val/
│   └── test/
└── data.yaml    ← YOLO training configuration
```

---

## 3. YOLO Label Format

Each `.txt` file has one line per object:

```
<class_id>  <cx>  <cy>  <width>  <height>
```

All values are **normalised 0–1** relative to image size.  Example for a
weed (class 0) centred at 40 % from left, 60 % from top, occupying
30 % width and 20 % height:

```
0  0.40  0.60  0.30  0.20
```

---

## 4. Labelling Your Own Images

Use [**LabelImg**](https://github.com/HumanSignal/labelImg) (free, offline):

```bash
pip install labelImg
labelImg
```

1. Open your image folder.
2. Set save format to **YOLO**.
3. Draw bounding boxes and assign class names.
4. Save – a `.txt` file is created alongside each image.

---

## 5. Training YOLOv8

Once the dataset directory is ready:

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=weed_detector
```

Training results (weights, curves, confusion matrix) are saved to
`runs/detect/weed_detector/`.

To use your trained model:

```bash
python demo/run_demo.py --mode yolo
# Then enter path: runs/detect/weed_detector/weights/best.pt
```

Or directly:

```bash
python vision/yolo_detection.py --weights runs/detect/weed_detector/weights/best.pt
```

---

## 6. Expected Training Performance (YOLOv8n, 50 epochs)

| Metric | Typical Value |
|--------|--------------|
| mAP@50 | 70–85 % |
| Precision | 75–88 % |
| Recall | 72–84 % |
| Training time (CPU) | ~2–4 hours |
| Training time (GPU) | ~15–30 min |

> **Tip:** Start with the pretrained `yolov8n.pt` – it already knows shapes,
> textures, and edges.  Fine-tuning on just ~500 labelled weed images can
> yield good results.
