"""
Simple script to convert CSV files with bounding boxes to YOLO format.
For beginners - no coding knowledge required!

CSV Format Expected:
filename, parasite_type, xmin, xmax, ymin, ymax
"""

import csv
import os
import shutil
from pathlib import Path
from PIL import Image
import random


def convert_csv_to_yolo(csv_file, images_folder, output_folder, train_split=0.8):
    """
    Convert CSV with bounding boxes to YOLO format and split into train/val.
    
    Args:
        csv_file: Path to your CSV file (e.g., "mp-idb-falciparum.csv")
        images_folder: Folder where your images are (e.g., "Falciparum/img")
        output_folder: Where to save the YOLO dataset (e.g., "Dataset")
        train_split: How much data for training (0.8 = 80% train, 20% validation)
    """
    
    print("=" * 60)
    print("Starting CSV to YOLO Conversion")
    print("=" * 60)
    
    # Step 1: Define class names (parasite types)
    # These are the classes we found in your CSV
    class_names = ['gam', 'ring', 'schi', 'tro']  # alphabetical order
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n✓ Found {len(class_names)} classes:")
    for idx, name in enumerate(class_names):
        print(f"  Class {idx}: {name}")
    
    # Step 2: Read the CSV file
    print(f"\n✓ Reading CSV file: {csv_file}")
    annotations = {}  # Store all annotations grouped by filename
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            parasite_type = row['parasite_type']
            xmin = int(row['xmin'])
            xmax = int(row['xmax'])
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])
            
            if filename not in annotations:
                annotations[filename] = []
            
            annotations[filename].append({
                'type': parasite_type,
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax
            })
    
    print(f"  Found {len(annotations)} unique images with annotations")
    
    # Step 3: Create output folders
    output_path = Path(output_folder)
    train_img_dir = output_path / "images" / "train"
    val_img_dir = output_path / "images" / "val"
    train_lbl_dir = output_path / "labels" / "train"
    val_lbl_dir = output_path / "labels" / "val"
    
    for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created output folders in: {output_folder}")
    
    # Step 4: Shuffle and split data
    all_filenames = list(annotations.keys())
    random.shuffle(all_filenames)
    
    split_index = int(len(all_filenames) * train_split)
    train_files = all_filenames[:split_index]
    val_files = all_filenames[split_index:]
    
    print(f"\n✓ Data split:")
    print(f"  Training: {len(train_files)} images ({train_split*100:.0f}%)")
    print(f"  Validation: {len(val_files)} images ({(1-train_split)*100:.0f}%)")
    
    # Step 5: Process each image
    print(f"\n✓ Converting images and labels...")
    
    processed_count = 0
    skipped_count = 0
    
    for filename in all_filenames:
        # Determine if this goes to train or val
        if filename in train_files:
            img_dir = train_img_dir
            lbl_dir = train_lbl_dir
            split_name = "train"
        else:
            img_dir = val_img_dir
            lbl_dir = val_lbl_dir
            split_name = "val"
        
        # Find the source image
        src_image_path = Path(images_folder) / filename
        
        if not src_image_path.exists():
            print(f"  ⚠ Warning: Image not found: {filename}")
            skipped_count += 1
            continue
        
        # Copy image
        dst_image_path = img_dir / filename
        shutil.copy2(src_image_path, dst_image_path)
        
        # Get image dimensions (needed for YOLO normalization)
        try:
            with Image.open(src_image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"  ⚠ Warning: Could not read image {filename}: {e}")
            skipped_count += 1
            continue
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in annotations[filename]:
            # Get class ID
            class_id = class_to_id.get(ann['type'])
            if class_id is None:
                print(f"  ⚠ Warning: Unknown class '{ann['type']}' in {filename}")
                continue
            
            # Convert from (xmin, xmax, ymin, ymax) to YOLO format
            # YOLO format: class_id x_center y_center width height (all normalized 0-1)
            x_center = ((ann['xmin'] + ann['xmax']) / 2) / img_width
            y_center = ((ann['ymin'] + ann['ymax']) / 2) / img_height
            width = abs(ann['xmax'] - ann['xmin']) / img_width
            height = abs(ann['ymax'] - ann['ymin']) / img_height
            
            # Ensure values are between 0 and 1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save label file
        label_filename = Path(filename).stem + ".txt"
        label_path = lbl_dir / label_filename
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        processed_count += 1
        
        # Show progress
        if processed_count % 50 == 0:
            print(f"  Processed {processed_count}/{len(all_filenames)} images...")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Successfully processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    
    # Step 6: Create class names file for reference
    class_names_file = output_path / "classes.txt"
    with open(class_names_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    
    print(f"\n✓ Saved class names to: {class_names_file}")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Your dataset is ready for training!")
    print("=" * 60)
    print(f"\nYour YOLO dataset is in: {output_folder}")
    print("\nNext steps:")
    print("1. Update config/data.yaml with the path to this dataset")
    print("2. Update the 'names' list in config/data.yaml with these classes:")
    for idx, name in enumerate(class_names):
        print(f"   - {name}")
    print("3. Run training!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CSV to YOLO Converter - For Plasmodium Detection")
    print("=" * 60)
    
    # CONFIGURATION - Change these paths to match your setup
    
    # Path to your CSV file
    csv_file = "MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis-master/Falciparum/mp-idb-falciparum.csv"
    
    # Folder where your images are
    images_folder = "MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis-master/Falciparum/img"
    
    # Where to save the YOLO dataset
    output_folder = "Dataset"
    
    # What percentage for training (0.8 = 80% train, 20% validation)
    train_split = 0.8
    
    print("\nConfiguration:")
    print(f"  CSV file: {csv_file}")
    print(f"  Images folder: {images_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Train/Val split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%")
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"\n❌ ERROR: CSV file not found: {csv_file}")
        print("Please update the 'csv_file' path in this script.")
        exit(1)
    
    if not os.path.exists(images_folder):
        print(f"\n❌ ERROR: Images folder not found: {images_folder}")
        print("Please update the 'images_folder' path in this script.")
        exit(1)
    
    input("\nPress ENTER to start conversion...")
    
    # Run the conversion
    convert_csv_to_yolo(csv_file, images_folder, output_folder, train_split)
    
    print("\n✅ All done! You can close this window.")
    input("Press ENTER to exit...")



