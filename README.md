## Plasmodium Detector - YOLOv8

End-to-end project to train and run a YOLO-based object detector for Plasmodium species on microscope slide images. Supports YOLO or COCO formatted datasets, handles class imbalance, logs to TensorBoard, exports to ONNX and TorchScript, provides CLI inference (images/folders/webcam), a minimal GUI, and a Docker image for inference.

### Requirements
- Python 3.9+
- GPU optional (recommended). Use `<<GPU_DEVICE>>` placeholder where noted.
- Ultralytics YOLOv8, PyTorch, OpenCV, Albumentations.

### Project Structure
```
.
├─ config/
│  ├─ data.yaml                  # dataset config (YOLO format)
│  └─ hyp.yaml                   # training hyperparameters
├─ src/
│  ├─ data_prep.py               # COCO→YOLO, imbalance augmentation
│  ├─ train.py                   # training entrypoint
│  ├─ eval.py                    # evaluation and report
│  ├─ infer.py                   # inference for images/folders/webcam
│  ├─ gui.py                     # minimal Tkinter GUI
│  ├─ export.py                  # export to ONNX / TorchScript
│  └─ utils.py                   # shared helpers (seed, logging, drawing)
├─ tests/
│  ├─ test_dataloader.py
│  └─ test_infer.py
├─ Dockerfile
├─ requirements.txt
└─ .gitignore
```

### Dataset Structure (YOLO format expected for training)
```
<<DATA_DIR>>/
├─ images/
│  ├─ train/  *.jpg|*.png
│  ├─ val/    *.jpg|*.png
│  └─ test/   *.jpg|*.png   # optional
└─ labels/
   ├─ train/  *.txt
   ├─ val/    *.txt
   └─ test/   *.txt         # optional
```
Each label file contains YOLO format lines: `class x_center y_center width height` normalized to [0,1].

If your data is in COCO format, use `src/data_prep.py --format coco` to convert to YOLO and optionally perform class-imbalance augmentation.

### Install
```
python -m venv .venv
.\.venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

Optional: login to Weights & Biases for experiment tracking:
```
pip install wandb
wandb login
```

### Configure
Edit `config/data.yaml` and replace placeholders:
- `<<DATA_DIR>>`: root folder of your dataset
- `names`: list your Plasmodium classes in order

Edit `config/hyp.yaml` as needed. Reasonable defaults are provided.

### Data Preparation
Convert COCO to YOLO and/or augment minority classes:
```
python src/data_prep.py \
  --input <<DATA_DIR>> \
  --output <<DATA_DIR>> \
  --format coco \
  --augment-minority \
  --min-samples-per-class 500
```
Notes:
- If your data is already YOLO, you can still run with `--format yolo` to compute class stats and optionally augment.
- Uses Albumentations for augmentation.

### Train
```
python src/train.py \
  --data config/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device <<GPU_DEVICE>> \
  --project "<<OUTPUT_DIR>>/runs" \
  --name plasmodium_yolov8n \
  --seed 42 \
  --use_wandb false
```
Recommendations:
- Start with `yolov8n.pt` or `yolov8s.pt`. Increase model size as needed.
- Typical ranges: `epochs=100-300`, `batch=16-64` depending on GPU memory.

### Evaluate
```
python src/eval.py \
  --weights "<<OUTPUT_DIR>>/runs/plasmodium_yolov8n/weights/best.pt" \
  --data config/data.yaml \
  --imgsz 640 \
  --device <<GPU_DEVICE>> \
  --save-json "<<OUTPUT_DIR>>/eval/metrics.json" \
  --save-csv "<<OUTPUT_DIR>>/eval/metrics.csv"
```
Outputs per-class precision/recall/mAP/F1 and a summary file.

### Inference (CLI)
Images or folder:
```
python src/infer.py \
  --weights "<<OUTPUT_DIR>>/runs/plasmodium_yolov8n/weights/best.pt" \
  --source "<<DATA_DIR>>/images/val" \
  --device <<GPU_DEVICE>> \
  --save-vis "<<OUTPUT_DIR>>/inference"
```
Webcam (0 is default camera):
```
python src/infer.py \
  --weights "<<OUTPUT_DIR>>/runs/plasmodium_yolov8n/weights/best.pt" \
  --webcam 0 \
  --device <<GPU_DEVICE>>
```

### GUI
```
python src/gui.py \
  --weights "<<OUTPUT_DIR>>/runs/plasmodium_yolov8n/weights/best.pt" \
  --device <<GPU_DEVICE>>
```
Use “Open Image” to load a file, or “Start Camera” for live detections.

### Export
```
python src/export.py \
  --weights "<<OUTPUT_DIR>>/runs/plasmodium_yolov8n/weights/best.pt" \
  --formats onnx torchscript \
  --imgsz 640 \
  --device cpu \
  --output "<<OUTPUT_DIR>>/exports"
```

### Docker (Inference)
Build:
```
docker build -t plasmodium-infer .
```
Run (CPU ONNXRuntime example):
```
docker run --rm -it ^
  -e MODEL_PATH=/app/model.onnx ^
  -v "<<OUTPUT_DIR>>/exports:/app" ^
  plasmodium-infer \
  python src/infer.py --onnx /app/model.onnx --source /app/sample.jpg --device cpu
```

### Reproducibility and Class Imbalance
- Seeds are set across numpy, torch, and python RNGs.
- Augmentation increases representation of minority classes.
- Optional focal loss via Ultralytics (`fl_gamma`) controlled in `config/hyp.yaml`.

### Unit Tests
Run:
```
pytest -q
```

### Notes
- Replace placeholders `<<DATA_DIR>>`, `<<OUTPUT_DIR>>`, `<<GPU_DEVICE>>` before running.
- If using COCO input, ensure `pycocotools` is installed (included in requirements).


