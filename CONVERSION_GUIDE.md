# 📖 Simple Guide: Converting Your Excel/CSV Data to YOLO Format

## What This Script Does

Your images have CSV files with bounding box information like this:
- `xmin`, `xmax`, `ymin`, `ymax` (coordinates in pixels)
- `parasite_type` (what kind of parasite)

The AI needs a different format called **YOLO format**:
- Text files (one per image)
- Format: `class_id x_center y_center width height` (all normalized 0-1)

This script converts everything automatically!

---

## Step-by-Step Instructions

### Step 1: Open Command Prompt
1. Press `Windows Key + R`
2. Type `cmd` and press Enter

### Step 2: Go to Your Project Folder
```bash
cd "C:\Users\kazzi\Desktop\AI Model-Plasmodium Detector-Mehrad"
```

### Step 3: Activate Your Virtual Environment
```bash
.\.venv\Scripts\activate
```
You should see `(.venv)` at the start of your line.

### Step 4: Run the Converter Script
```bash
python convert_csv_to_yolo.py
```

### Step 5: Press ENTER When Asked
The script will show you what it's going to do, then ask you to press ENTER to start.

### Step 6: Wait for Completion
You'll see messages like:
```
✓ Reading CSV file...
✓ Found 104 unique images
✓ Converting images and labels...
  Processed 50/104 images...
  Processed 100/104 images...
✓ Conversion complete!
```

---

## What Happens After Conversion?

The script creates this structure:

```
Dataset/
├── images/
│   ├── train/     (80% of your images - for training)
│   └── val/       (20% of your images - for testing)
├── labels/
│   ├── train/     (text files with bounding boxes)
│   └── val/       (text files with bounding boxes)
└── classes.txt    (list of parasite types)
```

---

## Understanding the Output

### Example: Original CSV Row
```csv
1305121398-0001-R_S.jpg,ring,919,887,76,67
```
**Meaning**: Image has a "ring" parasite at pixels (919, 887) to (76, 67)

### After Conversion: YOLO Format
**File**: `labels/train/1305121398-0001-R_S.txt`
```
1 0.459500 0.043850 0.026000 0.002250
```
**Meaning**: 
- `1` = class ID (ring)
- `0.459500` = x_center (45.95% from left)
- `0.043850` = y_center (4.38% from top)
- `0.026000` = width (2.6% of image width)
- `0.002250` = height (0.22% of image height)

---

## If You Have Multiple CSV Files

If you have other parasite types (Vivax, Malariae, Ovale), you can convert them too!

### Edit the Script
Open `convert_csv_to_yolo.py` in Notepad and change these lines:

```python
# For Vivax data:
csv_file = "MP-IDB-.../Vivax/mp-idb-vivax.csv"
images_folder = "MP-IDB-.../Vivax/img"
output_folder = "Dataset_Vivax"

# For Malariae:
csv_file = "MP-IDB-.../Malariae/mp-idb-malariae.csv"
images_folder = "MP-IDB-.../Malariae/img"
output_folder = "Dataset_Malariae"
```

Then run the script again for each dataset.

---

## Troubleshooting

### Problem: "CSV file not found"
**Solution**: The path is wrong. Check:
1. Is the folder name exactly correct? (case-sensitive)
2. Did you extract the ZIP file completely?

### Problem: "Images folder not found"
**Solution**: Make sure your images are in the `img` folder next to the CSV file.

### Problem: "No images found"
**Solution**: Check that your images are `.jpg` or `.png` files.

### Problem: "Skipped X images"
**Solution**: Some images mentioned in CSV might be missing. That's okay! The script will skip them and continue.

---

## Next Steps After Conversion

1. ✅ Your dataset is now ready in `Dataset/` folder
2. ✅ Update `config/data.yaml` (see instructions below)
3. ✅ Start training!

### Update config/data.yaml

Open `config/data.yaml` and change it to:

```yaml
path: "C:/Users/kazzi/Desktop/AI Model-Plasmodium Detector-Mehrad/Dataset"
train: images/train
val: images/val

names:
  - gam
  - ring
  - schi
  - tro
```

Save the file, then you're ready to train!

---

## What Are These Classes?

- **gam** = Gametocyte (sexual stage)
- **ring** = Ring stage (early stage)
- **schi** = Schizont (mature stage)
- **tro** = Trophozoite (growing stage)

These are different life stages of the Plasmodium parasite!

---

## Questions?

If something doesn't work:
1. Check the error message carefully
2. Make sure paths don't have spaces or special characters
3. Make sure you activated the virtual environment (`.venv`)
4. Try running the script again - sometimes it's just a temporary issue!



