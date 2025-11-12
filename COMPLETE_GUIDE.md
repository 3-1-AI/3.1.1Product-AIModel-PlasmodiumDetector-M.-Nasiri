# Plasmodium Detection System - Complete Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training the Model](#training-the-model)
5. [Testing & Evaluation](#testing--evaluation)
6. [Using the GUI](#using-the-gui)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)

---

## 🔬 Project Overview

AI-powered Plasmodium (malaria parasite) detection system using YOLOv8 object detection. The system can detect and classify four Plasmodium species:
- **Falciparum** (P. falciparum)
- **Vivax** (P. vivax)
- **Ovale** (P. ovale)
- **Malariae** (P. malariae)

### Features
✅ Real-time detection from microscope images or webcam  
✅ Advanced GUI with zoom and pan functionality  
✅ High accuracy object detection  
✅ Multiple dataset format support  
✅ Easy-to-use interface for medical professionals  

---

## 🚀 Installation

### Step 1: Clone/Navigate to Project
```bash
cd path/to/PlasmodiumDetector
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install CUDA PyTorch (for GPU acceleration)
```bash
# If you have NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📊 Dataset Preparation

### Converting DatasetNinja Format to YOLO

If you have the DatasetNinja format dataset (`mp-idb-DatasetNinja`):

**Step 1: Place Dataset**
```
project-root/
  ├── mp-idb-DatasetNinja/
  │   └── ds/
  │       ├── img/
  │       └── ann/
```

**Step 2: Run Conversion Script**
```bash
python convert_datasetninja_to_yolo.py
```

**What it does:**
- Extracts bounding boxes from bitmap masks
- Maps species names to class IDs
- Creates 80/20 train/validation split
- Generates YOLO-format labels

**Output:**
```
Dataset_DatasetNinja/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── classes.txt
```

### Dataset Configuration

The dataset config is located at `config/data_datasetninja.yaml`:

```yaml
path: "path/to/Dataset_DatasetNinja"

train: images/train
val: images/val

names:
  - falciparum  # Class 0
  - vivax       # Class 1
  - ovale       # Class 2
  - malariae    # Class 3
```

---

## 🎓 Training the Model

### Basic Training Command

```bash
python src/train.py \
  --data config/data_datasetninja.yaml \
  --epochs 100 \
  --batch 16 \
  --device cuda:0
```

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--data` | Path to data.yaml | Required | - |
| `--epochs` | Number of training epochs | 100 | 50-150 |
| `--batch` | Batch size | 16 | 16 (GPU), 8 (CPU) |
| `--device` | Device to use | cuda:0 | cuda:0 or cpu |
| `--model` | Base model | yolov8n.pt | yolov8n.pt (nano) |

### Training Output

Training results are saved to:
```
runs/plasmodium_yolov8/
  ├── weights/
  │   ├── best.pt      # Best model weights
  │   └── last.pt      # Last epoch weights
  ├── results.png      # Training metrics graphs
  ├── results.csv      # Metrics data
  └── confusion_matrix.png
```

### Monitoring Training

Training will display:
- Loss curves (box, cls, dfl)
- Precision and Recall
- mAP (mean Average Precision)
- Validation metrics

Expected training time:
- **GPU**: 1-3 hours
- **CPU**: 5-10 hours

---

## 🧪 Testing & Evaluation

### Method 1: View Training Results (Fastest)

Open the results from training:
```bash
# View graphs
runs/plasmodium_yolov8/results.png

# View metrics
runs/plasmodium_yolov8/results.csv
```

### Method 2: Run Evaluation Script

Get detailed per-class accuracy:
```bash
python src/eval.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --data config/data_datasetninja.yaml
```

Output:
- `eval/metrics.json` - Complete metrics
- `eval/metrics.csv` - Per-class mAP

### Method 3: Test on Single Image

```bash
python src/infer.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --source path/to/test_image.jpg \
  --data config/data_datasetninja.yaml
```

**Add `--save-vis results` to save annotated images:**
```bash
python src/infer.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --source Dataset_DatasetNinja/images/val/image.jpg \
  --data config/data_datasetninja.yaml \
  --save-vis results
```

### Method 4: Test on Folder

Process all images in a folder:
```bash
python src/infer.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --source Dataset_DatasetNinja/images/val \
  --data config/data_datasetninja.yaml \
  --save-vis results
```

### Understanding Metrics

**Key Metrics:**
- **mAP50-95**: Overall accuracy (0-1, higher is better)
- **mAP50**: Detection accuracy at 50% IoU threshold
- **Precision**: Percentage of correct predictions
- **Recall**: Percentage of parasites detected
- **F1-Score**: Harmonic mean of precision and recall

**Good Performance:**
- mAP50-95 > 0.7
- mAP50 > 0.85
- Precision & Recall > 0.8

---

## 🖥️ Using the GUI

### Launch GUI

```bash
python src/gui_advanced.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --data config/data_datasetninja.yaml
```

For CPU-only systems:
```bash
python src/gui_advanced.py \
  --weights runs/plasmodium_yolov8/weights/best.pt \
  --data config/data_datasetninja.yaml \
  --device cpu
```

### GUI Features

#### 🎛️ Control Panel (Left Side)

**📁 Open Image**
- Browse and select microscope images
- Supports: JPG, PNG, BMP, TIF
- Detection runs automatically

**💾 Save Result**
- Save annotated image with bounding boxes
- Choose output format (JPG/PNG)
- Original resolution preserved

**📷 Start/Stop Camera**
- Live detection from webcam or USB microscope
- Real-time inference
- Click again to stop

**🗑️ Clear**
- Reset display
- Clear current results

#### 🔍 Zoom Controls

**Buttons:**
- **➕** Zoom In (125% per click)
- **➖** Zoom Out (80% per click)
- **↺** Reset Zoom (100%)

**Mouse Wheel:**
- Scroll up = Zoom in
- Scroll down = Zoom out

**Zoom Range:** 10% to 500%

#### 🖱️ Pan (Move Image)

When zoomed in (>100%):
1. Click and hold left mouse button
2. Drag to move around image
3. Cursor changes to ✥ (move icon)
4. Automatically limits to boundaries

#### 📊 Detection Stats Panel

Shows real-time statistics:
- Total parasites detected
- Count per species
- Average confidence per class
- Available classes

#### 📈 Status Bar (Bottom)

Displays:
- Current operation status
- File names
- Detection counts
- Error messages

### GUI Workflow

1. **Launch GUI** with trained model
2. **Click "Open Image"** to select test image
3. **View detections** with bounding boxes and labels
4. **Use zoom** (scroll wheel) to examine details
5. **Click and drag** to pan around zoomed image
6. **Save results** if needed
7. **Test more images** or use camera for live detection

### Keyboard Shortcuts

- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan image (when zoomed)
- **ESC**: Close camera view (when camera active)

---

## 🔧 Troubleshooting

### Issue 1: CUDA Not Available

**Error:**
```
ValueError: Invalid CUDA 'device=0' requested
torch.cuda.is_available(): False
```

**Solutions:**

**Option A: Use CPU**
```bash
python src/train.py --data config/data.yaml --device cpu
```

**Option B: Install CUDA PyTorch**
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Option C: Check GPU**
```bash
nvidia-smi  # Should show GPU info
```

### Issue 2: Virtual Environment Path Issues

**Error:**
```
Fatal error in launcher: Unable to create process using...
```

**Solution:**
Use global Python or recreate venv:
```bash
deactivate
rm -r .venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue 3: Low Resolution in GUI

**Problem:** Blurry or low-quality display

**Solution:** DPI scaling is already fixed in `src/gui_advanced.py`. If still blurry:
1. Right-click `python.exe`
2. Properties → Compatibility
3. Check "Override high DPI scaling"
4. Select "System (Enhanced)"

### Issue 4: GUI Window Closes Immediately

**Solution:** Already fixed with `cv2.waitKey(0)` in `src/infer.py`

### Issue 5: Out of Memory During Training

**Error:**
```
CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--model yolov8n.pt`
- Close other programs
- Use CPU: `--device cpu`

### Issue 6: No Images Converted

**Check:**
1. Dataset structure is correct
2. `mp-idb-DatasetNinja/ds/img/` has images
3. `mp-idb-DatasetNinja/ds/ann/` has JSON files
4. Run conversion script with Python directly

### Issue 7: PIL Cannot Identify Image File

**Error when converting dataset:**
```
cannot identify image file
```

**Solution:** Already fixed in `convert_datasetninja_to_yolo.py` with zlib decompression

---

## 📁 Project Structure

```
PlasmodiumDetector/
│
├── config/
│   ├── data_datasetninja.yaml    # Dataset config for species detection
│   └── hyp.yaml                   # Hyperparameters
│
├── src/
│   ├── train.py                   # Training script
│   ├── eval.py                    # Evaluation script
│   ├── infer.py                   # Inference on images
│   ├── gui.py                     # Basic GUI
│   ├── gui_advanced.py            # Advanced GUI with zoom/pan
│   └── utils.py                   # Helper functions
│
├── convert_datasetninja_to_yolo.py  # Dataset conversion script
│
├── Dataset_DatasetNinja/          # Converted dataset (YOLO format)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── runs/                          # Training outputs
│   └── plasmodium_yolov8/
│       └── weights/
│           └── best.pt            # Trained model
│
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # Pre-trained YOLO model
│
└── COMPLETE_GUIDE.md             # This file
```

---

## 📝 Quick Reference Commands

### Dataset Conversion
```bash
python convert_datasetninja_to_yolo.py
```

### Training
```bash
# GPU
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --batch 16 --device cuda:0

# CPU
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --batch 8 --device cpu
```

### Testing Single Image
```bash
python src/infer.py --weights runs/plasmodium_yolov8/weights/best.pt --source test.jpg --data config/data_datasetninja.yaml --save-vis results
```

### Launch GUI
```bash
python src/gui_advanced.py --weights runs/plasmodium_yolov8/weights/best.pt --data config/data_datasetninja.yaml
```

### Evaluation
```bash
python src/eval.py --weights runs/plasmodium_yolov8/weights/best.pt --data config/data_datasetninja.yaml
```

---

## 🎯 Model Performance Tips

### Improving Accuracy

1. **More Data**
   - Collect more training images
   - Use data augmentation
   - Balance classes

2. **Longer Training**
   - Increase epochs: `--epochs 150`
   - Monitor for overfitting

3. **Larger Model**
   - Use `yolov8s.pt` or `yolov8m.pt`
   - Requires more GPU memory

4. **Hyperparameter Tuning**
   - Adjust learning rate
   - Modify augmentation settings
   - Change confidence threshold

### Best Practices

✅ Use consistent image quality  
✅ Ensure proper lighting in microscope images  
✅ Clean annotations (check converted labels)  
✅ Balance dataset across classes  
✅ Validate on separate test set  
✅ Monitor training metrics  

---

## 📊 Classes and Detection

### Supported Species

| Class ID | Species | Code |
|----------|---------|------|
| 0 | Plasmodium falciparum | falciparum |
| 1 | Plasmodium vivax | vivax |
| 2 | Plasmodium ovale | ovale |
| 3 | Plasmodium malariae | malariae |

### Detection Output

Each detection provides:
- **Bounding Box**: [x1, y1, x2, y2]
- **Class**: Species name
- **Confidence**: 0.0 - 1.0 (percentage)

Example:
```
falciparum: 0.95 (95% confidence)
vivax: 0.87 (87% confidence)
```

---

## 🤝 Usage Scenarios

### Scenario 1: Research Lab
1. Convert existing dataset
2. Train model with GPU
3. Evaluate on test set
4. Export model for deployment

### Scenario 2: Clinical Setting
1. Use pre-trained model
2. Run GUI on workstation
3. Connect USB microscope camera
4. Real-time detection during diagnosis

### Scenario 3: Field Testing
1. Load model on laptop
2. Process batch of images
3. Save annotated results
4. Generate statistics report

---

## 📚 Additional Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Dataset Information
- Original dataset: MP-IDB (Malaria Parasite Image Database)
- DatasetNinja format with bitmap annotations
- Converted to YOLO bounding box format

### Model Information
- Base: YOLOv8 (Ultralytics)
- Task: Object Detection
- Architecture: CNN-based
- Input: RGB images (640x640 default)
- Output: Bounding boxes + class probabilities

---

## 🐛 Reporting Issues

If you encounter problems:

1. Check this troubleshooting guide
2. Verify installation steps
3. Check Python and package versions
4. Review error messages carefully
5. Test with CPU mode first

---

## ✅ Success Checklist

Before sharing your model:

- [ ] Dataset converted successfully
- [ ] Training completed without errors
- [ ] mAP > 0.7 on validation set
- [ ] GUI launches and loads model
- [ ] Can detect parasites in test images
- [ ] Zoom and pan work properly
- [ ] Camera mode functional (if needed)
- [ ] Documentation updated

---

## 📞 Summary

This guide covers the complete workflow from dataset preparation to model deployment. Follow the steps in order:

1. ✅ Install dependencies
2. ✅ Convert dataset to YOLO format
3. ✅ Train model (GPU recommended)
4. ✅ Evaluate performance
5. ✅ Use GUI for inference

**Training time:** ~1-3 hours (GPU) or ~5-10 hours (CPU)  
**Expected accuracy:** mAP50-95 > 0.7  
**Use case:** Automated malaria parasite detection  

---

*Guide created: 2025*  
*Last updated: November 2025*  
*For: Plasmodium Detection AI Model Project*

