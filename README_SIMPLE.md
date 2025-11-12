# 🦟 Plasmodium Detector - Simple Guide for Beginners

## 🎯 What This Project Does

Automatically detects malaria parasites in microscope images using AI.

---

## 📊 Your Complete Workflow (3 Simple Steps)

```
┌─────────────────┐
│  Step 1         │
│  CONVERT DATA   │
│                 │
│  CSV + Images   │
│       ↓         │
│  YOLO Format    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Step 2         │
│  TRAIN AI       │
│                 │
│  Learn from     │
│  83 images      │
│  (2-4 hours)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Step 3         │
│  USE AI         │
│                 │
│  Detect         │
│  parasites in   │
│  new images     │
└─────────────────┘
```

---

## 🚀 Quick Commands (Copy-Paste These)

### 1. Setup (One Time Only)
```bash
cd "C:\Users\kazzi\Desktop\AI Model-Plasmodium Detector-Mehrad"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
⏱️ Takes: 10-20 minutes

---

### 2. Convert Your Data
```bash
.\.venv\Scripts\activate
python convert_csv_to_yolo.py
```
⏱️ Takes: 1-3 minutes

---

### 3. Check Everything is Ready
```bash
python check_setup.py
```
✅ Should show all green checkmarks

---

### 4. Train the AI

**If you have NVIDIA GPU:**
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --device cuda:0 --project runs --name falciparum_v1 --seed 42
```
⏱️ Takes: 20-40 minutes

**If you only have CPU:**
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu --project runs --name falciparum_v1 --seed 42
```
⏱️ Takes: 2-4 hours (be patient!)

---

### 5. Test Your AI

**Test on one image:**
```bash
python src/infer.py --weights runs/falciparum_v1/weights/best.pt --source "MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis-master/Falciparum/img/1305121398-0001-R_S.jpg" --device cpu --save-vis results/test
```

**Test with GUI (easy!):**
```bash
python src/gui.py --weights runs/falciparum_v1/weights/best.pt --device cpu --data config/data.yaml
```

---

## 📁 Your Data Structure

### Before Conversion (What you have):
```
MP-IDB-.../Falciparum/
├── img/
│   └── *.jpg files (104 images)
└── mp-idb-falciparum.csv (coordinates)
```

### After Conversion (What AI needs):
```
Dataset/
├── images/
│   ├── train/ (83 images - 80%)
│   └── val/   (21 images - 20%)
└── labels/
    ├── train/ (83 text files)
    └── val/   (21 text files)
```

---

## 🎓 Understanding Your Classes

Your dataset has 4 types of parasites:

| ID | Name | Description |
|----|------|-------------|
| 0 | **gam** | Gametocyte - banana shaped 🌙 |
| 1 | **ring** | Ring stage - small ring 💍 |
| 2 | **schi** | Schizont - big with dots 🎯 |
| 3 | **tro** | Trophozoite - blue blob 🔵 |

These are different life stages of *Plasmodium falciparum* malaria parasite.

---

## 📊 Understanding Training Results

When training shows:
```
mAP50-95: 0.52
```

This means:
- **0.0 - 0.3**: ❌ Poor (need more training)
- **0.3 - 0.5**: ⚠️ Fair (usable)
- **0.5 - 0.7**: ✅ Good
- **0.7 - 0.9**: ✅✅ Very good
- **0.9 - 1.0**: ✅✅✅ Excellent

---

## 🔧 Common Issues & Quick Fixes

### "Python not recognized"
```bash
# Install Python from python.org
# Check "Add to PATH" during installation
```

### "No module named 'torch'"
```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### "CUDA out of memory"
```bash
# Use smaller batch size:
--batch 4
```

### Training too slow
```bash
# Normal on CPU! Options:
# 1. Get GPU (NVIDIA)
# 2. Use fewer epochs: --epochs 20
# 3. Be patient (2-4 hours is normal)
```

---

## 📖 Detailed Guides

- **BEGINNER_TUTORIAL.md** - Complete step-by-step guide with explanations
- **CONVERSION_GUIDE.md** - Details about CSV to YOLO conversion
- **QUICK_START.md** - Visual guide with examples

---

## 🎯 Your Checklist

Before training:
- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Packages installed (`pip install -r requirements.txt`)
- [ ] Data converted (`python convert_csv_to_yolo.py`)
- [ ] Setup verified (`python check_setup.py`)

Ready to train? ✅
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --device cpu --project runs --name my_model
```

---

## 💡 Tips for Success

1. **Start small**: Test with 20 epochs first to make sure it works
2. **Be patient**: CPU training takes hours - start it and do something else
3. **Check results**: Look at `runs/my_model/results.png` to see training progress
4. **Test often**: Try your model on images to see if it's working
5. **Don't worry about errors**: They're normal! Read the message and fix it

---

## 🆘 Need Help?

1. Run `python check_setup.py` to diagnose issues
2. Read the error message carefully
3. Check the detailed guides in this folder
4. Google the error (many people have solved the same issues)

---

## 🎉 What You'll Achieve

After following this guide:
- ✅ Trained AI model that detects malaria parasites
- ✅ Can analyze new microscope images automatically
- ✅ Understand AI training workflow
- ✅ Have a useful tool for malaria research

**Good luck! 🔬🚀**

---

*Created by: Mehrad Nasiri | Project: Plasmodium Detector using YOLOv8*



