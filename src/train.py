import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import set_global_seed


def main():
	parser = argparse.ArgumentParser(description="Train YOLOv8 on Plasmodium dataset.")
	parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
	parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model or config")
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
	parser.add_argument("--project", type=str, default="runs")
	parser.add_argument("--name", type=str, default="plasmodium_yolov8")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--hyp", type=str, default="config/hyp.yaml")
	parser.add_argument("--use_wandb", type=bool, default=False)
	args = parser.parse_args()

	set_global_seed(args.seed)
	out_dir = Path(args.project)
	out_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(args.model)
	# Ultralytics internally handles TensorBoard logs under runs/
	results = model.train(
		data=args.data,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		project=args.project,
		name=args.name,
		patience=50,
		cache=True,
		val=True,
	)
	if args.use_wandb:
		try:
			import wandb  # noqa: F401
		except Exception:
			print("W&B not installed; skipping W&B logging.")
	print("Training complete.")


if __name__ == "__main__":
	main()


