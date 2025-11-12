# 🔬 Plasmodium Detection System

AI-powered malaria parasite detection using YOLOv8 deep learning. Detect and classify four Plasmodium species from microscope images in real-time.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 🎯 **High Accuracy Detection** - YOLOv8-based object detection
- 🦠 **4 Species Classification** - Falciparum, Vivax, Ovale, Malariae
- 🖥️ **Modern GUI** - Interactive interface with zoom and pan
- 📷 **Live Camera Support** - Real-time detection from USB microscope
- ⚡ **GPU Accelerated** - Fast inference with CUDA support
- 💾 **Easy Export** - Save annotated results

## 🎬 Demo

```
[GUI Interface]
- Left Panel: Controls (Open, Save, Camera, Zoom)
- Center: Detection view with bounding boxes
- Right: Statistics (counts, confidence scores)
```

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Dataset
```bash
python convert_datasetninja_to_yolo.py
```

### 3. Train
```bash
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --device cuda:0
```

### 4. Launch GUI
```bash
python src/gui_advanced.py --weights runs/plasmodium_yolov8/weights/best.pt --data config/data_datasetninja.yaml
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| mAP50-95 | > 0.7 |
| mAP50 | > 0.85 |
| Inference Speed (GPU) | < 1s per image |
| Training Time (GPU) | 1-3 hours |

## 🖼️ Supported Species

| Species | Description |
|---------|-------------|
| 🦠 **P. falciparum** | Most deadly malaria parasite |
| 🦠 **P. vivax** | Most common outside Africa |
| 🦠 **P. ovale** | Found mainly in Africa |
| 🦠 **P. malariae** | Causes chronic infections |

## 🎮 GUI Features

- **Zoom**: Mouse wheel or ➕/➖ buttons (10% - 500%)
- **Pan**: Click and drag when zoomed in
- **Live Detection**: Connect USB microscope for real-time analysis
- **Statistics**: Real-time counts and confidence scores
- **Export**: Save annotated images in JPG/PNG

## 📁 Project Structure

```
PlasmodiumDetector/
├── src/
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation
│   ├── infer.py              # Inference
│   └── gui_advanced.py       # Advanced GUI
├── config/
│   └── data_datasetninja.yaml  # Dataset config
├── convert_datasetninja_to_yolo.py  # Dataset converter
├── COMPLETE_GUIDE.md         # Full documentation
└── QUICK_REFERENCE.md        # Quick commands
```

## 📚 Documentation

- 📖 **[Complete Guide](COMPLETE_GUIDE.md)** - Detailed step-by-step instructions
- ⚡ **[Quick Reference](QUICK_REFERENCE.md)** - Essential commands cheat sheet
- 🔧 **Troubleshooting** - Common issues and solutions (in Complete Guide)

## 🛠️ Tech Stack

- **Framework**: PyTorch 2.9
- **Model**: YOLOv8 (Ultralytics)
- **GUI**: Tkinter + PIL + OpenCV
- **Image Processing**: OpenCV, Pillow
- **Data Format**: YOLO (converted from DatasetNinja)

## 📋 Requirements

- Python 3.13+
- NVIDIA GPU (recommended) or CPU
- 8GB+ RAM
- Windows/Linux/MacOS

## 🎯 Use Cases

### 🏥 Clinical Diagnosis
- Real-time parasite detection during microscopy
- Automated counting and classification
- Quality control for manual diagnosis

### 🔬 Research Labs
- Dataset analysis and annotation
- Model performance evaluation
- Algorithm comparison

### 🌍 Field Deployment
- Portable diagnostic tool
- Batch image processing
- Remote area screening

## 📊 Dataset

- **Source**: MP-IDB (Malaria Parasite Image Database)
- **Format**: DatasetNinja → YOLO conversion
- **Images**: Microscope blood smear images
- **Annotations**: Bounding boxes with species labels
- **Split**: 80% train / 20% validation

## ⚙️ Training Options

```bash
# GPU Training (Recommended)
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --batch 16 --device cuda:0

# CPU Training
python src/train.py --data config/data_datasetninja.yaml --epochs 100 --batch 8 --device cpu

# Custom Settings
python src/train.py --data config/data.yaml --epochs 150 --batch 32 --model yolov8m.pt
```

## 🧪 Testing & Evaluation

```bash
# Single Image
python src/infer.py --weights runs/.../best.pt --source test.jpg --data config/data_datasetninja.yaml

# Batch Processing
python src/infer.py --weights runs/.../best.pt --source images_folder/ --data config/data_datasetninja.yaml --save-vis results

# Evaluation Metrics
python src/eval.py --weights runs/.../best.pt --data config/data_datasetninja.yaml
```

## 🔍 Model Output

Each detection provides:
- **Bounding Box**: Parasite location [x1, y1, x2, y2]
- **Species**: falciparum / vivax / ovale / malariae
- **Confidence**: 0.0 - 1.0 (detection certainty)

Example:
```
Detection: falciparum (95.3%)
Box: [342, 156, 398, 212]
```

## 🐛 Troubleshooting

**CUDA not available?**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
```bash
python src/train.py --batch 8 --device cpu
```

**Low resolution GUI?**
- DPI awareness automatically enabled
- Fixed in `src/gui_advanced.py`

See **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** for more solutions.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional species support
- Model optimization
- UI/UX enhancements
- Documentation translations
- Performance benchmarks

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **MP-IDB** - Dataset source
- **DatasetNinja** - Dataset format and tools
- **PyTorch Team** - Deep learning framework

## 📞 Support

- 📖 Read **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** for detailed instructions
- ⚡ Check **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** for common commands
- 🐛 Review troubleshooting section for common issues

## 🎓 Citation

If you use this project in your research, please cite:

```
Plasmodium Detection System
AI-powered malaria parasite detection using YOLOv8
2025
```

## 📈 Future Enhancements

- [ ] Multi-stage classification (life cycle stages)
- [ ] Mobile app deployment
- [ ] Cloud-based inference API
- [ ] Automated reporting system
- [ ] Integration with lab management systems
- [ ] Support for additional image formats
- [ ] 3D visualization of detection results

---

**⭐ Star this repository if you find it useful!**

**📧 Questions? See documentation or open an issue.**

**🚀 Ready to start? Follow the [Quick Start](#-quick-start) guide above!**

