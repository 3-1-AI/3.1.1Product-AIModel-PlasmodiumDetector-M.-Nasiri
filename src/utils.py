import os
import random
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import cv2


def set_global_seed(seed: int = 42) -> None:
	"""
	Ensure reproducibility as much as possible.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows: List[List[Any]], header: Optional[List[str]] = None) -> None:
	import csv
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		if header:
			writer.writerow(header)
		writer.writerows(rows)


def draw_detections(
	image_bgr: np.ndarray,
	boxes: List[Tuple[int, int, int, int]],
	classes: List[int],
	scores: List[float],
	class_names: List[str],
) -> np.ndarray:
	"""
	Draw bounding boxes with labels on a copy of the image.
	"""
	out = image_bgr.copy()
	for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
		color = (0, 180, 255)
		cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
		label = f"{class_names[cls]} {score:.2f}"
		(txt_w, txt_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		cv2.rectangle(out, (x1, y1 - txt_h - baseline - 4), (x1 + txt_w + 2, y1), color, -1)
		cv2.putText(out, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	return out


def load_class_names_from_data_yaml(data_yaml_path: str | Path) -> List[str]:
	import yaml
	with open(data_yaml_path, "r", encoding="utf-8") as f:
		y = yaml.safe_load(f)
	names = y.get("names", [])
	if isinstance(names, dict):
		# convert COCO-style {id: name} to list
		names = [names[k] for k in sorted(names.keys())]
	return names


