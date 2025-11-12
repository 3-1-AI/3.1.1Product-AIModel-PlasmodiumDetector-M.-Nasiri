# 🚀 Quick Start Guide - For Complete Beginners

## Your Journey from CSV to Trained AI Model

```
Your Current Data          Convert             Ready for AI         Train          Trained Model
    (CSV files)      →    to YOLO       →      (Dataset/)      →   the AI    →     (best.pt)
   
   image1.jpg                                    images/
   image2.jpg             Run the                ├── train/
   data.csv          →    converter       →      └── val/        →   Wait...   →   🎉 Done!
   (xmin,xmax,              script                labels/
    ymin,ymax)                                   ├── train/
                                                 └── val/
```

---

## ⚡ Super Quick Commands (Copy & Paste)

### 1️⃣ Convert Your Data
```bash
cd "C:\Users\kazzi\Desktop\AI Model-Plasmodium Detector-Mehrad"
.\.venv\Scripts\activate
python convert_csv_to_yolo.py
```

### 2️⃣ Train the AI
```bash
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu --project runs --name falciparum_detector --seed 42
```
**Note**: Change `--device cpu` to `--device cuda:0` if you have NVIDIA GPU

### 3️⃣ Test Your Model
```bash
python src/infer.py --weights runs/falciparum_detector/weights/best.pt --source "MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis-master/Falciparum/img" --device cpu --save-vis results/detections
```

### 4️⃣ Try the GUI
```bash
python src/gui.py --weights runs/falciparum_detector/weights/best.pt --device cpu --data config/data.yaml
```

---

## 📊 What Each Parasite Type Means

Your dataset has 4 types of parasites (life stages of Plasmodium falciparum):

| Class ID | Name | Full Name | Description | Visual |
|----------|------|-----------|-------------|--------|
| 0 | **gam** | Gametocyte | Sexual reproduction stage | 🌙 Banana/crescent shaped |
| 1 | **ring** | Ring Stage | Early stage, just infected | 💍 Small ring/signet shape |
| 2 | **schi** | Schizont | Mature, about to burst | 🎯 Multiple nuclei, large |
| 3 | **tro** | Trophozoite | Growing stage | 🔵 Round, blue/purple blob |

---

## 🎯 Your Dataset Overview

Based on your CSV file, you have:

- **Total Images**: 104 microscope images
- **Total Parasites**: ~1,299 individual parasite annotations
- **Image Format**: JPG images (around 2000x1500 pixels)
- **Parasite Types**: 4 classes (gam, ring, schi, tro)

After conversion:
- **Training Set**: ~83 images (80%)
- **Validation Set**: ~21 images (20%)

---

## 🕐 How Long Things Take

### On CPU (no GPU):
- **Conversion**: 1-2 minutes ✅ Fast!
- **Training (50 epochs)**: 2-4 hours ⏳ Slow but works
- **Inference (1 image)**: 2-5 seconds ✅ Fast!

### On GPU (NVIDIA with CUDA):
- **Conversion**: 1-2 minutes ✅ Fast!
- **Training (50 epochs)**: 15-30 minutes ⚡ Much faster!
- **Inference (1 image)**: 0.1-0.5 seconds ⚡ Super fast!

---

## 📈 Understanding Training Output

When training runs, you'll see something like this:

```
Epoch 1/50: 100%|███████████| 83/83 [01:45<00:00]
      Class    Images  Instances      P      R  mAP50  mAP50-95
        all        21        260   0.35   0.42   0.38      0.21
        gam        21         15   0.40   0.50   0.45      0.25
       ring        21        180   0.35   0.40   0.35      0.18
       schi        21         50   0.38   0.44   0.40      0.22
        tro        21         15   0.30   0.35   0.32      0.19
```

### What These Numbers Mean:

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **P** (Precision) | "When AI says it found a parasite, how often is it correct?" | > 0.7 (70%) |
| **R** (Recall) | "Of all parasites in the image, how many did AI find?" | > 0.7 (70%) |
| **mAP50** | Overall accuracy at 50% overlap | > 0.6 (60%) |
| **mAP50-95** | Strict accuracy (50-95% overlap) | > 0.4 (40%) |

### What You'll See During Training:

**Early epochs (1-10)**: Numbers are low (0.2-0.4) - AI is just learning
**Middle epochs (10-30)**: Numbers improving (0.4-0.6) - AI is getting better
**Late epochs (30-50)**: Numbers plateau (0.6-0.8) - AI has learned

---

## 🎨 Visual Example: Before and After

### Before (CSV Format):
```csv
filename,parasite_type,xmin,xmax,ymin,ymax
image1.jpg,ring,919,887,76,67
image1.jpg,schi,1246,1498,106,113
```

### After (YOLO Format):
**File: labels/train/image1.txt**
```
1 0.903000 0.071500 0.016000 0.004500
2 0.622000 0.109500 0.126000 0.003500
```

### What the Converter Does:
1. ✅ Reads your CSV
2. ✅ Opens each image to get its size
3. ✅ Converts pixel coordinates to percentages (0-1)
4. ✅ Calculates center point and box size
5. ✅ Assigns class IDs (gam=0, ring=1, schi=2, tro=3)
6. ✅ Splits data into train/val
7. ✅ Copies images and saves text files

---

## 🔍 Checking If Conversion Worked

After running the converter, check these:

### ✅ Files Created:
```
Dataset/
├── images/
│   ├── train/  (should have ~83 .jpg files)
│   └── val/    (should have ~21 .jpg files)
├── labels/
│   ├── train/  (should have ~83 .txt files)
│   └── val/    (should have ~21 .txt files)
└── classes.txt (should list: gam, ring, schi, tro)
```

### ✅ Manually Check One File:

1. Open: `Dataset/labels/train/1305121398-0001-R_S.txt`
2. You should see lines like: `1 0.459500 0.043850 0.026000 0.002250`
3. Each line is one parasite
4. First number is class ID (0-3)
5. Next 4 numbers are between 0 and 1

If you see this, conversion worked! ✅

---

## 🛠️ Troubleshooting Common Issues

### Issue: "Python is not recognized"
**Fix**: Python not installed or not in PATH. Reinstall Python and check "Add to PATH"

### Issue: "No module named 'PIL'"
**Fix**: Run `pip install Pillow` in your activated venv

### Issue: Training crashes with "CUDA out of memory"
**Fix**: Lower batch size: `--batch 4` or `--batch 2`

### Issue: Very low accuracy after training
**Possible reasons**:
1. Not enough epochs (try `--epochs 100`)
2. Parasites are very small/hard to see
3. Dataset too small (104 images is on the low side)

**Solutions**:
- Train longer (100-200 epochs)
- Use a bigger model (`yolov8s.pt` instead of `yolov8n.pt`)
- Augment data (the script already splits it randomly)

### Issue: "Path not found" errors
**Fix**: Use forward slashes in paths: `C:/Users/...` not `C:\Users\...`

---

## 🎓 Next Steps After Your First Training

1. **Evaluate Your Model**:
   ```bash
   python src/eval.py --weights runs/falciparum_detector/weights/best.pt --data config/data.yaml --device cpu --save-json results/metrics.json --save-csv results/metrics.csv
   ```

2. **Test on New Images**:
   - Put test images in a folder
   - Run inference
   - Check if detections are correct

3. **Improve the Model**:
   - Train longer (100 epochs)
   - Try bigger model (yolov8s.pt)
   - Add more data if possible

4. **Export for Deployment**:
   ```bash
   python src/export.py --weights runs/falciparum_detector/weights/best.pt --formats onnx --output exports/
   ```

---

## 📝 Checklist: Am I Ready to Train?

- [ ] Python 3.9+ installed
- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Virtual environment activated (`(.venv)` shows in command prompt)
- [ ] Requirements installed (`pip install -r requirements.txt` done)
- [ ] CSV converted to YOLO (`Dataset/` folder has images and labels)
- [ ] `config/data.yaml` points to `Dataset/` folder
- [ ] `config/data.yaml` has correct class names (gam, ring, schi, tro)

If all checked ✅ → You're ready to train! 🚀

---

## 🆘 Need Help?

If something doesn't work:

1. **Read the error message** - it usually tells you what's wrong
2. **Check file paths** - make sure they're correct
3. **Check you activated venv** - should see `(.venv)` in prompt
4. **Try a simpler test first** - run on 1 image before 100 images
5. **Google the error** - many people have solved similar issues

Remember: **It's okay if it doesn't work first try!** Debugging is part of learning. 💪

---

Good luck with your malaria detection AI! 🔬🦟



