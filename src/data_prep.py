import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import cv2
import albumentations as A

from utils import ensure_dir, load_class_names_from_data_yaml


def coco_to_yolo(input_dir: Path, output_dir: Path) -> None:
	"""
	Convert a COCO dataset with splits (train/val[/test]) to YOLO format into output_dir.
	Assumes COCO annotations in input_dir/annotations/{instances_train.json, instances_val.json, ...}
	and images in input_dir/{train, val, test}.
	"""
	try:
		from pycocotools.coco import COCO
	except Exception as e:
		raise RuntimeError("pycocotools is required for COCO conversion. Please install it.") from e

	ann_dir = input_dir / "annotations"
	for split in ["train", "val", "test"]:
		json_path = ann_dir / f"instances_{split}.json"
		img_dir = input_dir / split
		if not json_path.exists() or not img_dir.exists():
			if split in ["train", "val"]:
				raise FileNotFoundError(f"Missing COCO files for split={split}: {json_path} or {img_dir}")
			else:
				continue

		coco = COCO(str(json_path))
		img_ids = coco.getImgIds()
		out_img_dir = ensure_dir(output_dir / "images" / split)
		out_lbl_dir = ensure_dir(output_dir / "labels" / split)

		# Copy images and write YOLO labels
		for img_id in img_ids:
			img_info = coco.loadImgs([img_id])[0]
			file_name = img_info["file_name"]
			src_img = img_dir / file_name
			dst_img = out_img_dir / file_name
			dst_img.parent.mkdir(parents=True, exist_ok=True)
			if not dst_img.exists():
				shutil.copy2(src_img, dst_img)

			width = img_info["width"]
			height = img_info["height"]
			ann_ids = coco.getAnnIds(imgIds=[img_id])
			anns = coco.loadAnns(ann_ids)

			lines: List[str] = []
			for a in anns:
				if a.get("iscrowd", 0) == 1:
					continue
				x, y, w, h = a["bbox"]
				xc = (x + w / 2) / width
				yc = (y + h / 2) / height
				wn = w / width
				hn = h / height
				cid = int(a["category_id"]) - 1  # assuming ids start at 1 and continuous
				lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

			lbl_path = out_lbl_dir / (Path(file_name).stem + ".txt")
			with open(lbl_path, "w", encoding="utf-8") as f:
				f.write("\n".join(lines))


def compute_class_distribution(labels_dir: Path, num_classes: int) -> Counter:
	counts = Counter()
	for txt in labels_dir.rglob("*.txt"):
		with open(txt, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				cls_id = int(line.split()[0])
				if 0 <= cls_id < num_classes:
					counts[cls_id] += 1
	return counts


def build_augmenter() -> A.Compose:
	return A.Compose([
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.1),
		A.RandomBrightnessContrast(p=0.3),
		A.HueSaturationValue(p=0.3),
		A.GaussianBlur(blur_limit=3, p=0.2),
		A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
	], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def load_yolo_label(path: Path) -> Tuple[List[List[float]], List[int]]:
	boxes: List[List[float]] = []
	classes: List[int] = []
	if not path.exists():
		return boxes, classes
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) != 5:
				continue
			cls = int(parts[0]); x, y, w, h = map(float, parts[1:])
			boxes.append([x, y, w, h])
			classes.append(cls)
	return boxes, classes


def save_yolo_label(path: Path, boxes: List[List[float]], classes: List[int]) -> None:
	lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in zip(classes, boxes)]
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def augment_minority_classes(
	root_dir: Path,
	split: str,
	min_samples_per_class: int,
	class_names: List[str],
) -> None:
	img_dir = root_dir / "images" / split
	lbl_dir = root_dir / "labels" / split
	if not img_dir.exists() or not lbl_dir.exists():
		return

	num_classes = len(class_names)
	dist = compute_class_distribution(lbl_dir, num_classes)
	augmenter = build_augmenter()

	for cls_id in range(num_classes):
		current = dist.get(cls_id, 0)
		if current >= min_samples_per_class:
			continue
		to_add = min_samples_per_class - current
		# collect samples containing this class
		candidates: List[Path] = []
		for txt in lbl_dir.glob("*.txt"):
			boxes, classes = load_yolo_label(txt)
			if cls_id in classes and len(boxes) > 0:
				candidates.append(txt)
		if not candidates:
			continue

		idx = 0
		while to_add > 0 and candidates:
			src_lbl = candidates[idx % len(candidates)]
			src_img = img_dir / (src_lbl.stem + ".jpg")
			if not src_img.exists():
				src_img = img_dir / (src_lbl.stem + ".png")
				if not src_img.exists():
					idx += 1
					continue

			image = cv2.imread(str(src_img))
			if image is None:
				idx += 1
				continue
			h, w = image.shape[:2]
			boxes, classes = load_yolo_label(src_lbl)
			# Albumentations expects bboxes in yolo normalized, convert labels to match
			labels = [class_names[c] for c in classes]
			aug = augmenter(image=image, bboxes=boxes, class_labels=labels)
			aug_img = aug["image"]
			aug_boxes = aug["bboxes"]
			aug_classes = classes  # labels preserved order; safe enough for balancing usage

			# write new files
			out_base = f"{src_lbl.stem}_aug_{to_add}"
			out_img = img_dir / f"{out_base}.jpg"
			out_lbl = lbl_dir / f"{out_base}.txt"
			cv2.imwrite(str(out_img), aug_img)
			save_yolo_label(out_lbl, aug_boxes, aug_classes)

			to_add -= 1
			idx += 1


def main():
	parser = argparse.ArgumentParser(description="Data preparation: COCO->YOLO conversion and class imbalance augmentation.")
	parser.add_argument("--input", type=str, required=True, help="Input dataset root (COCO or YOLO).")
	parser.add_argument("--output", type=str, required=True, help="Output dataset root (YOLO). Can be same as input for in-place.")
	parser.add_argument("--format", type=str, choices=["coco", "yolo"], default="yolo", help="Input format.")
	parser.add_argument("--augment-minority", action="store_true", help="Augment minority classes to reach min samples.")
	parser.add_argument("--min-samples-per-class", type=int, default=500)
	parser.add_argument("--data-yaml", type=str, default="config/data.yaml", help="Data yaml to read class names.")
	args = parser.parse_args()

	input_dir = Path(args.input)
	output_dir = Path(args.output)
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.format == "coco":
		coco_to_yolo(input_dir, output_dir)
	else:
		# if already YOLO, optionally copy structure if output != input
		if input_dir.resolve() != output_dir.resolve():
			if output_dir.exists():
				shutil.rmtree(output_dir)
			shutil.copytree(input_dir, output_dir)

	if args.augment_minority:
		class_names = load_class_names_from_data_yaml(args.data_yaml)
		for split in ["train"]:
			augment_minority_classes(output_dir, split, args.min_samples_per_class, class_names)

	print("Data preparation complete.")


if __name__ == "__main__":
	main()


