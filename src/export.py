import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import ensure_dir


def main():
	parser = argparse.ArgumentParser(description="Export YOLO model to ONNX / TorchScript.")
	parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"], help="Any of: onnx, torchscript")
	parser.add_argument("--output", type=str, default="<<OUTPUT_DIR>>/exports")
	args = parser.parse_args()

	out_dir = ensure_dir(args.output)
	model = YOLO(args.weights)
	for fmt in args.formats:
		print(f"Exporting to {fmt}...")
		res_path = model.export(format=fmt, imgsz=args.imgsz, device=args.device, opset=12)
		# Move/copy exported file(s) to output dir
		p = Path(res_path)
		dst = out_dir / p.name
		if p.exists():
			dst.write_bytes(p.read_bytes())
		print(f"Saved: {dst}")
	print("Export complete.")


if __name__ == "__main__":
	main()


