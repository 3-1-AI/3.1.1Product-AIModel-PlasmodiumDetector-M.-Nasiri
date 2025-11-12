"""
Setup Checker - Verify everything is ready for training
Run this before training to catch common issues!
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a nice header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def check_python_version():
    """Check if Python version is 3.9+"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} (Good!)")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} is too old!")
        print("   Please install Python 3.9 or newer")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLOv8',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'yaml': 'PyYAML'
    }
    
    missing = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            missing.append(name)
    
    if missing:
        print("\n❌ Missing packages! Run this command:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print_success("All packages installed!")
        return True

def check_dataset_structure():
    """Check if dataset is properly structured"""
    print("\n📁 Checking dataset structure...")
    
    required_folders = [
        "Dataset/images/train",
        "Dataset/images/val",
        "Dataset/labels/train",
        "Dataset/labels/val"
    ]
    
    all_exist = True
    for folder in required_folders:
        path = Path(folder)
        if path.exists():
            # Count files
            if 'images' in folder:
                files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            else:
                files = list(path.glob("*.txt"))
            
            if len(files) > 0:
                print_success(f"{folder}/ exists ({len(files)} files)")
            else:
                print_warning(f"{folder}/ exists but is EMPTY!")
                all_exist = False
        else:
            print_error(f"{folder}/ NOT found")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Dataset not ready!")
        print("   Run the converter first: python convert_csv_to_yolo.py")
        return False
    else:
        print_success("Dataset structure looks good!")
        return True

def check_config_file():
    """Check if config/data.yaml is properly configured"""
    print("\n⚙️  Checking configuration file...")
    
    config_path = Path("config/data.yaml")
    
    if not config_path.exists():
        print_error("config/data.yaml NOT found!")
        return False
    
    print_success("config/data.yaml exists")
    
    # Read and check contents
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check path
        if 'path' not in config:
            print_error("'path' not defined in config/data.yaml")
            return False
        
        dataset_path = config['path']
        if "<<DATA_DIR>>" in dataset_path:
            print_error("'path' still has placeholder <<DATA_DIR>>")
            print("   Update it to point to your Dataset folder")
            return False
        
        print_success(f"Dataset path: {dataset_path}")
        
        # Check if path exists
        if not Path(dataset_path).exists():
            print_warning(f"Path '{dataset_path}' does not exist!")
            print("   Make sure the path is correct")
        
        # Check classes
        if 'names' not in config:
            print_error("'names' not defined in config/data.yaml")
            return False
        
        classes = config['names']
        if len(classes) == 0:
            print_error("No classes defined in 'names'")
            return False
        
        print_success(f"Found {len(classes)} classes: {', '.join(classes)}")
        
        return True
        
    except Exception as e:
        print_error(f"Error reading config file: {e}")
        return False

def check_gpu():
    """Check if GPU is available"""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU available: {gpu_name}")
            print("   You can use: --device cuda:0")
            return True
        else:
            print_warning("No GPU detected")
            print("   Training will use CPU (slower)")
            print("   Use: --device cpu")
            return False
    except Exception as e:
        print_warning(f"Could not check GPU: {e}")
        return False

def estimate_training_time():
    """Estimate training time"""
    print("\n⏱️  Estimating training time...")
    
    # Count training images
    train_img_path = Path("Dataset/images/train")
    if train_img_path.exists():
        num_images = len(list(train_img_path.glob("*.jpg")) + list(train_img_path.glob("*.png")))
        
        if num_images > 0:
            print(f"   Training images: {num_images}")
            
            # Rough estimates
            try:
                import torch
                if torch.cuda.is_available():
                    time_per_epoch = (num_images / 100) * 1.5  # rough estimate
                    device = "GPU"
                else:
                    time_per_epoch = (num_images / 100) * 8  # CPU is slower
                    device = "CPU"
                
                print(f"   On {device}:")
                print(f"   - 50 epochs: ~{int(time_per_epoch * 50 / 60)} minutes")
                print(f"   - 100 epochs: ~{int(time_per_epoch * 100 / 60)} minutes")
                
            except:
                print("   (Could not estimate time)")

def print_next_steps(all_checks_passed):
    """Print what to do next"""
    print_header("Summary")
    
    if all_checks_passed:
        print_success("All checks passed! You're ready to train! 🚀")
        print("\nNext steps:")
        print("\n1️⃣  Start training with this command:")
        print("   python src/train.py --data config/data.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --device cpu --project runs --name my_model --seed 42")
        print("\n   (Change --device cpu to --device cuda:0 if you have GPU)")
        print("\n2️⃣  Wait for training to complete (check estimates above)")
        print("\n3️⃣  Your trained model will be at: runs/my_model/weights/best.pt")
        print("\n4️⃣  Test it with:")
        print("   python src/infer.py --weights runs/my_model/weights/best.pt --source <image_path> --device cpu")
    else:
        print_error("Some checks failed! Fix the issues above before training.")
        print("\nCommon fixes:")
        print("- Missing packages: pip install -r requirements.txt")
        print("- No dataset: python convert_csv_to_yolo.py")
        print("- Wrong config: Edit config/data.yaml")

def main():
    """Run all checks"""
    print_header("🔍 Setup Checker for Plasmodium Detector")
    print("This script will verify your setup before training.")
    
    checks = []
    
    # Run all checks
    checks.append(("Python version", check_python_version()))
    checks.append(("Required packages", check_packages()))
    checks.append(("Dataset structure", check_dataset_structure()))
    checks.append(("Configuration file", check_config_file()))
    checks.append(("GPU availability", check_gpu()))
    
    # Estimate time (doesn't affect pass/fail)
    estimate_training_time()
    
    # Check if critical checks passed
    critical_checks = checks[:-1]  # All except GPU check
    all_passed = all(result for name, result in critical_checks)
    
    # Print summary
    print_next_steps(all_passed)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ Setup complete! Happy training! 🎓")
    else:
        print("❌ Please fix the issues above, then run this script again.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress ENTER to exit...")



