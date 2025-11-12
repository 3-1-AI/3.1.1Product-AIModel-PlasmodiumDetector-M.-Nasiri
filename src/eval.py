import argparse
from pathlib import Path
from typing import List

from ultralytics import YOLO
from utils import write_json, write_csv, load_class_names_from_data_yaml


def main():
	parser = argparse.ArgumentParser(description="Evaluate trained YOLO model and save per-class metrics.")
	parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
	parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--save-json", type=str, default="eval/metrics.json")
	parser.add_argument("--save-csv", type=str, default="eval/metrics.csv")
	args = parser.parse_args()

	model = YOLO(args.weights)
	res = model.val(data=args.data, imgsz=args.imgsz, device=args.device, plots=True)

	# Ultralytics returns rich metrics; extract per-class stats.
	# The 'results' object contains fields like: metrics.box.map, maps (per-class mAP50-95), etc.
	# For per-class precision/recall, use res.box.pr (precision-recall arrays) and res.box.names
	class_names: List[str] = load_class_names_from_data_yaml(args.data)

	# Attempt to collect per-class metrics; fallback to available fields
	maps = getattr(res, "maps", None)  # list per-class mAP50-95
	# The new API exposes res.results dict-like. We'll compute a simple report.
	summary = {
		"metrics/mAP50-95": float(getattr(res, "box", getattr(res, "metrics", object())).map if hasattr(getattr(res, "box", object()), "map") else getattr(res, "map", float("nan"))),
		"metrics/mAP50": float(getattr(getattr(res, "box", object()), "map50", float("nan"))),
		"metrics/precision": float(getattr(getattr(res, "box", object()), "mp", float("nan"))),
		"metrics/recall": float(getattr(getattr(res, "box", object()), "mr", float("nan"))),
	}

	rows = []
	for i, name in enumerate(class_names):
		m = float(maps[i]) if maps is not None and i < len(maps) else float("nan")
		rows.append([i, name, m])

	out_json = Path(args.save_json)
	out_csv = Path(args.save_csv)
	out_json.parent.mkdir(parents=True, exist_ok=True)
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	write_json(out_json, {"summary": summary, "per_class": [{"id": r[0], "name": r[1], "mAP50-95": r[2]} for r in rows]})
	write_csv(out_csv, rows, header=["class_id", "class_name", "mAP50-95"])
	print(f"Saved metrics to {out_json} and {out_csv}")


if __name__ == "__main__":
	main()


