# 🎯 START HERE - Your First Steps

## Welcome! 👋

You're about to train an AI to detect malaria parasites in microscope images. This guide will get you started in **3 simple steps**.

---

## ⚡ Ultra-Quick Start (5 Commands)

If you know what you're doing, just run these:

```bash
# 1. Setup
cd "C:\Users\kazzi\Desktop\AI Model-Plasmodium Detector-Mehrad"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Convert data
python convert_csv_to_yolo.py

# 3. Train (takes 2-4 hours on CPU)
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --device cpu --project runs --name my_model
```

Done! Your model will be at: `runs/my_model/weights/best.pt`

---

## 📚 If You're New to Coding

Don't worry! I've created detailed guides for you:

### 1️⃣ **BEGINNER_TUTORIAL.md** ← START HERE!
- Complete step-by-step guide
- Explains every command
- Troubleshooting tips
- **Perfect for complete beginners**

### 2️⃣ **README_SIMPLE.md**
- Quick overview of the project
- Simple workflow diagram
- Common issues & fixes

### 3️⃣ **CONVERSION_GUIDE.md**
- Details about converting your CSV data
- How to handle multiple datasets
- Understanding label formats

### 4️⃣ **QUICK_START.md**
- Visual guide with examples
- Expected outputs
- Performance tips

### 5️⃣ **COMMANDS_CHEATSHEET.txt**
- All commands in one place
- Copy-paste ready
- Parameter explanations

---

## 🎯 Your Project Overview

### What You Have:
- 104 microscope images of blood with malaria parasites
- CSV file with bounding box coordinates (xmin, xmax, ymin, ymax)
- 4 parasite types: gam, ring, schi, tro

### What You'll Create:
- AI model that automatically detects and classifies parasites
- Works on new images you haven't seen before
- Can process images in seconds

### What You'll Learn:
- How to prepare data for AI
- How to train a detection model
- How to evaluate and use the model
- Basic AI/ML workflow

---

## 📁 Important Files in This Project

| File | What It Does |
|------|--------------|
| `convert_csv_to_yolo.py` | Converts your CSV data to AI format |
| `check_setup.py` | Verifies everything is installed correctly |
| `src/train.py` | Trains the AI model |
| `src/infer.py` | Uses the model to detect parasites |
| `src/gui.py` | Simple graphical interface |
| `config/data.yaml` | Dataset configuration |

---

## ⏱️ Time Estimates

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| Setup & Install | 15-20 min | 15-20 min |
| Convert Data | 2-3 min | 2-3 min |
| Train (50 epochs) | 2-4 hours | 20-40 min |
| Test Model | 2-5 sec/image | 0.2-0.5 sec/image |

**Total time to first working model**: 
- With CPU: ~3-5 hours
- With GPU: ~30-60 minutes

---

## 🚦 Step-by-Step Roadmap

```
┌─────────────────────────────────────────────┐
│ Step 1: Setup (15 min)                      │
│ ✓ Install Python                            │
│ ✓ Create virtual environment                │
│ ✓ Install packages                          │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│ Step 2: Prepare Data (3 min)                │
│ ✓ Run converter script                      │
│ ✓ Check dataset structure                   │
│ ✓ Verify configuration                      │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│ Step 3: Train Model (2-4 hours)             │
│ ✓ Start training                            │
│ ✓ Monitor progress                          │
│ ✓ Wait for completion                       │
└─────────────┬───────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────┐
│ Step 4: Test & Use (5 min)                  │
│ ✓ Evaluate accuracy                         │
│ ✓ Test on images                            │
│ ✓ Try GUI interface                         │
└─────────────────────────────────────────────┘
```

---

## ✅ Pre-Flight Checklist

Before starting, make sure you have:

- [ ] Windows computer (Windows 10 or 11)
- [ ] At least 8 GB RAM (16 GB recommended)
- [ ] 10 GB free disk space
- [ ] Internet connection (to download packages)
- [ ] 3-4 hours of time (for training)
- [ ] *Optional:* NVIDIA GPU (makes training 10x faster)

---

## 🎬 What to Do Right Now

### **Option A: I'm a complete beginner**
1. Open `BEGINNER_TUTORIAL.md`
2. Read from the start
3. Follow step-by-step
4. Don't skip steps!

### **Option B: I have some coding experience**
1. Open `README_SIMPLE.md` for overview
2. Run `python check_setup.py` to verify setup
3. Open `COMMANDS_CHEATSHEET.txt` for quick reference
4. Start with the 5 commands at the top of this file

### **Option C: I'm in a hurry**
```bash
# Just copy-paste this entire block:
cd "C:\Users\kazzi\Desktop\AI Model-Plasmodium Detector-Mehrad"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python convert_csv_to_yolo.py
python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --device cpu --project runs --name quick
```
Wait 2-4 hours, then your model is ready!

---

## 🆘 If Something Goes Wrong

1. **First**: Run `python check_setup.py` to diagnose
2. **Second**: Check `BEGINNER_TUTORIAL.md` → Troubleshooting section
3. **Third**: Read the error message carefully (it often tells you the fix)
4. **Fourth**: Google the error (many people have solved the same issues)

---

## 📊 Expected Results

After training, you should see:

**Good Model** (after 50 epochs):
- mAP50-95: 0.4 - 0.6
- Detects most parasites
- Some false positives

**Great Model** (after 100 epochs, bigger model):
- mAP50-95: 0.6 - 0.8
- Detects almost all parasites
- Few false positives

**Perfect Model** (rare, needs lots of data):
- mAP50-95: 0.8+
- Detects all parasites accurately
- Very few errors

---

## 💡 Pro Tips

1. **Start with a test run**: Use `--epochs 2` first to make sure everything works
2. **Train overnight**: Start training before bed, it'll be done in the morning
3. **Save your work**: Copy `runs/` folder to backup your trained models
4. **Try different models**: yolov8n (fast) vs yolov8s (accurate)
5. **Look at results**: Check `runs/<name>/results.png` to see training graphs

---

## 🎓 Learning Resources

Included in this project:
- ✅ `BEGINNER_TUTORIAL.md` - Complete guide
- ✅ `CONVERSION_GUIDE.md` - Data preparation details
- ✅ `QUICK_START.md` - Quick reference
- ✅ `COMMANDS_CHEATSHEET.txt` - All commands
- ✅ `README_SIMPLE.md` - Simple overview

External resources (free):
- YOLOv8 Documentation: https://docs.ultralytics.com/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- fast.ai Course: https://course.fast.ai/

---

## 🎯 Success Criteria

You'll know you succeeded when:
- ✅ Training completes without errors
- ✅ mAP50-95 score is above 0.4
- ✅ Model detects parasites in test images
- ✅ GUI shows detections with bounding boxes

---

## 🚀 Ready to Start?

Choose your path:

**🟢 Beginner (Never coded before)**
→ Open `BEGINNER_TUTORIAL.md` and start reading

**🟡 Intermediate (Some experience)**
→ Open `README_SIMPLE.md` for overview, then run commands

**🔴 Advanced (Know what you're doing)**
→ Open `COMMANDS_CHEATSHEET.txt` and go!

---

## 📞 Project Information

- **Project**: Plasmodium Detector using YOLOv8
- **Author**: Mehrad Nasiri
- **Purpose**: Automatic malaria parasite detection in microscope images
- **Dataset**: MP-IDB Falciparum (104 images, 4 classes)
- **Technology**: Python, PyTorch, YOLOv8, OpenCV

---

## 🎉 Final Words

Training AI might seem complicated, but you've got everything you need:
- ✅ The code
- ✅ The data
- ✅ Detailed guides
- ✅ Step-by-step instructions

**Just follow the guides and be patient!**

Most people succeed on their first try if they:
1. Read instructions carefully
2. Don't skip steps
3. Be patient (training takes time)
4. Check for errors and fix them

**You got this! 💪**

---

**→ Next step: Open `BEGINNER_TUTORIAL.md` and start! 🚀**

---

*Last updated: November 2025*



