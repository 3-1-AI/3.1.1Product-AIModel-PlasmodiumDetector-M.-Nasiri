# Plasmodium Detection - Quick Reference Card

## 🚀 Quick Start (3 Steps)

### 1️⃣ Convert Dataset
```bash
python convert_datasetninja_to_yolo.py
```

### 2️⃣ Train Model
```bash
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --batch 16 --device cuda:0
```

### 3️⃣ Launch GUI
```bash
python src/gui_advanced.py --weights runs/plasmodium_yolov8/weights/best.pt --data config/data_datasetninja.yaml
```

---

## 📋 Essential Commands

| Task | Command |
|------|---------|
| **Install packages** | `pip install -r requirements.txt` |
| **Install CUDA PyTorch** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| **Check CUDA** | `python -c "import torch; print(torch.cuda.is_available())"` |
| **Train (GPU)** | `python src/train.py --data config/data_datasetninja.yaml --device cuda:0` |
| **Train (CPU)** | `python src/train.py --data config/data_datasetninja.yaml --device cpu --batch 8` |
| **Test image** | `python src/infer.py --weights runs/.../best.pt --source image.jpg --data config/data_datasetninja.yaml` |
| **Evaluate** | `python src/eval.py --weights runs/.../best.pt --data config/data_datasetninja.yaml` |
| **GUI** | `python src/gui_advanced.py --weights runs/.../best.pt --data config/data_datasetninja.yaml` |

---

## 🖥️ GUI Controls

| Action | Method |
|--------|--------|
| **Open image** | Click "📁 Open Image" button |
| **Zoom in** | Scroll up OR click ➕ |
| **Zoom out** | Scroll down OR click ➖ |
| **Reset zoom** | Click ↺ |
| **Pan (move)** | Click and drag (when zoomed) |
| **Save result** | Click "💾 Save Result" |
| **Live camera** | Click "📷 Start Camera" |
| **Clear** | Click "🗑️ Clear" |

---

## 🔧 Common Fixes

| Problem | Solution |
|---------|----------|
| CUDA not available | Add `--device cpu` to command |
| Out of memory | Reduce batch: `--batch 8` or `--batch 4` |
| Low GUI resolution | Already fixed with DPI awareness |
| venv path error | Use global Python: `deactivate` first |
| Window closes fast | Fixed with `waitKey(0)` |

---

## 📊 Classes

- **0**: falciparum
- **1**: vivax
- **2**: ovale
- **3**: malariae

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `runs/.../weights/best.pt` | Trained model |
| `runs/.../results.png` | Training graphs |
| `config/data_datasetninja.yaml` | Dataset config |
| `COMPLETE_GUIDE.md` | Full documentation |

---

## 🎯 Good Model Performance

- ✅ mAP50-95 > 0.7
- ✅ mAP50 > 0.85
- ✅ Precision > 0.8
- ✅ Recall > 0.8

---

## ⏱️ Expected Times

- Training: 1-3 hours (GPU) / 5-10 hours (CPU)
- Inference: <1 second per image (GPU) / 2-5 seconds (CPU)
- Dataset conversion: 2-5 minutes

---

**See COMPLETE_GUIDE.md for detailed instructions**

