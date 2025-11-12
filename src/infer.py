import argparse
from pathlib import Path
from typing import Optional
import time
import cv2
import numpy as np
from ultralytics import YOLO
from utils import ensure_dir, load_class_names_from_data_yaml, draw_detections


def run_on_source(model: YOLO, source: str, device: str, save_vis: Optional[str], class_names: list[str]) -> None:
	paths = []
	p = Path(source)
	if p.is_dir():
		for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"):
			paths.extend(p.glob(ext))
	elif p.is_file():
		paths.append(p)
	else:
		raise FileNotFoundError(f"Source not found: {source}")

	save_dir = ensure_dir(save_vis) if save_vis else None
	for img_path in paths:
		res = model.predict(source=str(img_path), device=device, conf=0.25, verbose=False)
		# parse first result
		r = res[0]
		xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
		cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
		conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))

		image = cv2.imread(str(img_path))
		boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in xyxy]
		out = draw_detections(image, boxes, cls.tolist(), conf.tolist(), class_names)
		cv2.imshow("Detections", out)
		cv2.waitKey(1)
		if save_dir:
			cv2.imwrite(str(save_dir / f"{img_path.stem}_det.jpg"), out)
	cv2.destroyAllWindows()


def run_on_webcam(model: YOLO, cam_index: int, device: str, class_names: list[str]) -> None:
	cap = cv2.VideoCapture(cam_index)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open webcam index {cam_index}")
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		res = model.predict(source=frame, device=device, conf=0.25, verbose=False)
		r = res[0]
		xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
		cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
		conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
		boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in xyxy]
		out = draw_detections(frame, boxes, cls.tolist(), conf.tolist(), class_names)
		cv2.imshow("Live", out)
		if cv2.waitKey(1) & 0xFF == 27:
			break
	cap.release()
	cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description="YOLO Inference on images/folder or webcam.")
	parser.add_argument("--weights", type=str, help="Path to .pt weights")
	parser.add_argument("--onnx", type=str, default=None, help="Optional ONNX path for onnxruntime inference (not required; .pt preferred)")
	parser.add_argument("--source", type=str, help="Image file or folder")
	parser.add_argument("--webcam", type=int, default=None, help="Webcam index")
	parser.add_argument("--device", type=str, default="<<GPU_DEVICE>>")
	parser.add_argument("--data", type=str, default="config/data.yaml", help="To fetch class names")
	parser.add_argument("--save-vis", type=str, default=None, help="Folder to save visualizations")
	args = parser.parse_args()

	class_names = load_class_names_from_data_yaml(args.data)
	if args.onnx:
		# Keep the same API path for simplicity: Ultralytics can also load exported formats
		model = YOLO(args.onnx)
	else:
		if not args.weights:
			raise ValueError("Provide --weights for .pt model or --onnx for ONNX model.")
		model = YOLO(args.weights)

	if args.webcam is not None:
		run_on_webcam(model, args.webcam, args.device, class_names)
	elif args.source:
		run_on_source(model, args.source, args.device, args.save_vis, class_names)
	else:
		raise ValueError("Provide --source for images/folder or --webcam for live camera.")


if __name__ == "__main__":
	main()


