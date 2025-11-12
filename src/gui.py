import argparse
import threading
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import cv2
from ultralytics import YOLO
from utils import load_class_names_from_data_yaml, draw_detections


class App:
	def __init__(self, root: tk.Tk, model_path: str, device: str, class_names: list[str]):
		self.root = root
		self.root.title("Plasmodium Detector")
		self.model = YOLO(model_path)
		self.device = device
		self.class_names = class_names
		self.cam_running = False
		self.cam_thread = None

		btn_frame = tk.Frame(root)
		btn_frame.pack(padx=10, pady=10)
		tk.Button(btn_frame, text="Open Image", command=self.open_image).grid(row=0, column=0, padx=5)
		tk.Button(btn_frame, text="Start Camera", command=self.start_camera).grid(row=0, column=1, padx=5)
		tk.Button(btn_frame, text="Stop Camera", command=self.stop_camera).grid(row=0, column=2, padx=5)

	def open_image(self):
		path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif")])
		if not path:
			return
		img = cv2.imread(path)
		res = self.model.predict(source=img, device=self.device, conf=0.25, verbose=False)[0]
		xyxy = res.boxes.xyxy.cpu().numpy()
		cls = res.boxes.cls.cpu().numpy().astype(int)
		conf = res.boxes.conf.cpu().numpy()
		boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in xyxy]
		out = draw_detections(img, boxes, cls.tolist(), conf.tolist(), self.class_names)
		cv2.imshow("Detections", out)
		cv2.waitKey(1)

	def start_camera(self):
		if self.cam_running:
			return
		self.cam_running = True
		self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
		self.cam_thread.start()

	def stop_camera(self):
		self.cam_running = False

	def _cam_loop(self):
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			self.cam_running = False
			return
		while self.cam_running:
			ret, frame = cap.read()
			if not ret:
				break
			res = self.model.predict(source=frame, device=self.device, conf=0.25, verbose=False)[0]
			xyxy = res.boxes.xyxy.cpu().numpy()
			cls = res.boxes.cls.cpu().numpy().astype(int)
			conf = res.boxes.conf.cpu().numpy()
			boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in xyxy]
			out = draw_detections(frame, boxes, cls.tolist(), conf.tolist(), self.class_names)
			cv2.imshow("Live", out)
			if cv2.waitKey(1) & 0xFF == 27:
				break
		cap.release()
		cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description="Minimal GUI for Plasmodium Detector")
	parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
	parser.add_argument("--device", type=str, default="<<GPU_DEVICE>>")
	parser.add_argument("--data", type=str, default="config/data.yaml")
	args = parser.parse_args()

	class_names = load_class_names_from_data_yaml(args.data)
	root = tk.Tk()
	app = App(root, args.weights, args.device, class_names)
	root.protocol("WM_DELETE_WINDOW", app.stop_camera)
	root.mainloop()


if __name__ == "__main__":
	main()




