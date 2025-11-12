"""
Convert DatasetNinja format (mp-idb-DatasetNinja) to YOLO format.
Extracts bounding boxes from bitmap masks and maps Plasmodium species to classes.
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import base64
import io
import random


def decode_bitmap(bitmap_data, origin):
    """Decode base64 bitmap to numpy array and get bounding box."""
    import zlib
    
    # Decode base64
    img_data = base64.b64decode(bitmap_data)
    
    # Try to decompress if it's zlib compressed
    try:
        img_data = zlib.decompress(img_data)
    except:
        pass  # Not compressed or already decompressed
    
    # Try to open as image
    try:
        bitmap_img = Image.open(io.BytesIO(img_data))
        bitmap_array = np.array(bitmap_img)
    except:
        # If not an image, treat as raw binary mask
        bitmap_array = np.frombuffer(img_data, dtype=np.uint8)
    
    # Get mask dimensions
    if len(bitmap_array.shape) == 3:
        # If RGB, take first channel or convert to grayscale
        mask = bitmap_array[:, :, 0] if bitmap_array.shape[2] > 0 else bitmap_array
    else:
        mask = bitmap_array
    
    # Find non-zero pixels (the mask)
    rows, cols = np.where(mask > 0)
    
    if len(rows) == 0:
        return None
    
    # Get bounding box in mask coordinates
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Convert to image coordinates using origin
    x_origin, y_origin = origin
    xmin = x_origin + min_col
    ymin = y_origin + min_row
    xmax = x_origin + max_col
    ymax = y_origin + max_row
    
    return xmin, ymin, xmax, ymax


def map_tags_to_class(tags):
    """Map life stage tags to class names."""
    tag_mapping = {
        'gametocyte stage': 'gam',
        'ring stage': 'ring',
        'schizont stage': 'schi',
        'trophozoite stage': 'tro'
    }
    
    for tag in tags:
        tag_name = tag.get('name', '')
        if tag_name in tag_mapping:
            return tag_mapping[tag_name]
    
    # If no stage tag found, try to infer from filename
    return None


def get_class_from_filename(filename):
    """Extract class from filename (fallback method)."""
    # Filenames like "1305121398-0001-R_S.jpg" where R=ring, S=schizont, etc.
    mapping = {
        'G': 'gam',
        'R': 'ring',
        'S': 'schi',
        'T': 'tro'
    }
    
    # Extract the part after last dash before extension
    basename = Path(filename).stem
    parts = basename.split('-')
    if len(parts) >= 3:
        stage_part = parts[-1]
        # Take first letter if multiple stages (e.g., "R_S" -> "R")
        stage_letter = stage_part.split('_')[0][0] if stage_part else None
        return mapping.get(stage_letter)
    
    return None


def convert_datasetninja_to_yolo(
    input_folder,
    output_folder,
    train_split=0.8
):
    """
    Convert DatasetNinja format to YOLO format.
    
    Args:
        input_folder: Path to mp-idb-DatasetNinja folder
        output_folder: Where to save YOLO dataset
        train_split: Fraction for training (0.8 = 80% train, 20% val)
    """
    
    print("=" * 60)
    print("DatasetNinja to YOLO Converter")
    print("=" * 60)
    
    input_path = Path(input_folder)
    img_dir = input_path / "ds" / "img"
    ann_dir = input_path / "ds" / "ann"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    
    # Define class names (species)
    class_names = ['falciparum', 'vivax', 'ovale', 'malariae']
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n✓ Classes: {class_names}")
    
    # Create output folders
    output_path = Path(output_folder)
    train_img_dir = output_path / "images" / "train"
    val_img_dir = output_path / "images" / "val"
    train_lbl_dir = output_path / "labels" / "train"
    val_lbl_dir = output_path / "labels" / "val"
    
    for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output folder: {output_folder}")
    
    # Get all annotation files
    ann_files = sorted(ann_dir.glob("*.json"))
    print(f"✓ Found {len(ann_files)} annotations")
    
    # Shuffle and split
    random.shuffle(ann_files)
    split_idx = int(len(ann_files) * train_split)
    train_files = ann_files[:split_idx]
    val_files = ann_files[split_idx:]
    
    print(f"✓ Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process files
    processed = 0
    skipped = 0
    total_objects = 0
    
    print("\n✓ Converting...")
    
    for ann_file in ann_files:
        # Determine split
        if ann_file in train_files:
            img_out_dir = train_img_dir
            lbl_out_dir = train_lbl_dir
        else:
            img_out_dir = val_img_dir
            lbl_out_dir = val_lbl_dir
        
        # Get corresponding image
        img_name = ann_file.stem  # Remove .json
        img_path = img_dir / img_name
        
        if not img_path.exists():
            print(f"⚠ Warning: Image not found: {img_name}")
            skipped += 1
            continue
        
        # Read annotation
        with open(ann_file, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
        
        # Get image dimensions
        img_width = ann_data['size']['width']
        img_height = ann_data['size']['height']
        
        # Process objects
        yolo_lines = []
        
        for obj in ann_data.get('objects', []):
            # Get bitmap data
            bitmap = obj.get('bitmap')
            if not bitmap:
                continue
            
            try:
                # Decode bitmap and get bounding box
                bbox = decode_bitmap(bitmap['data'], bitmap['origin'])
                if bbox is None:
                    continue
                
                xmin, ymin, xmax, ymax = bbox
                
                # Get class from object's classTitle (species name)
                class_name = obj.get('classTitle', '').lower()
                
                if class_name is None or class_name not in class_to_id:
                    continue
                
                class_id = class_to_id[class_name]
                
                # Convert to YOLO format (normalized)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Clamp values
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                total_objects += 1
                
            except Exception as e:
                print(f"⚠ Error processing object in {img_name}: {e}")
                continue
        
        if not yolo_lines:
            # Skip images with no valid annotations
            skipped += 1
            continue
        
        # Copy image
        shutil.copy2(img_path, img_out_dir / img_name)
        
        # Save label file
        label_filename = Path(img_name).stem + ".txt"
        label_path = lbl_out_dir / label_filename
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        
        processed += 1
        
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(ann_files)}...")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Processed: {processed} images")
    print(f"  Skipped: {skipped} images")
    print(f"  Total objects: {total_objects}")
    
    # Save class names
    class_file = output_path / "classes.txt"
    with open(class_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    
    print(f"\n✓ Saved class names to: {class_file}")
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Dataset ready for training")
    print("=" * 60)
    print(f"\nDataset location: {output_folder}")
    print("\nNext steps:")
    print("1. Update config/data.yaml with this path")
    print("2. Train your model!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DatasetNinja to YOLO Converter")
    print("=" * 60)
    
    # Configuration
    input_folder = "mp-idb-DatasetNinja"
    output_folder = "Dataset_DatasetNinja"
    train_split = 0.8
    
    print(f"\nConfiguration:")
    print(f"  Input: {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Train/Val split: {train_split*100:.0f}%/{(1-train_split)*100:.0f}%")
    
    # Check if input exists
    if not Path(input_folder).exists():
        print(f"\n❌ ERROR: Input folder not found: {input_folder}")
        exit(1)
    
    input("\nPress ENTER to start conversion...")
    
    # Run conversion
    convert_datasetninja_to_yolo(
        input_folder,
        output_folder,
        train_split
    )
    
    print("\n✅ All done!")
    input("Press ENTER to exit...")

