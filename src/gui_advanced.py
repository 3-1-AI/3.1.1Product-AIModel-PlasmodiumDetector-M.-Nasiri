import argparse
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from utils import load_class_names_from_data_yaml, draw_detections

# Fix DPI scaling on Windows
try:
	from ctypes import windll
	windll.shcore.SetProcessDpiAwareness(1)
except:
	pass


class PlasmodiumDetectorGUI:
	def __init__(self, root: tk.Tk, model_path: str, device: str, class_names: list[str]):
		self.root = root
		self.root.title("Plasmodium Detector - AI Model")
		self.root.geometry("1200x800")
		self.root.configure(bg="#2c3e50")
		
		self.model = YOLO(model_path)
		self.device = device
		self.class_names = class_names
		self.cam_running = False
		self.cam_thread = None
		self.current_image = None
		self.current_detections = None
		self.zoom_level = 1.0
		self.pan_x = 0
		self.pan_y = 0
		self.drag_start_x = 0
		self.drag_start_y = 0
		self.is_dragging = False
		
		# Title
		title_frame = tk.Frame(root, bg="#34495e", height=60)
		title_frame.pack(fill=tk.X, padx=10, pady=10)
		tk.Label(
			title_frame, 
			text="🔬 Plasmodium Detection System", 
			font=("Arial", 20, "bold"),
			bg="#34495e",
			fg="white"
		).pack(pady=15)
		
		# Main content frame
		content_frame = tk.Frame(root, bg="#2c3e50")
		content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		
		# Left panel - Controls
		left_panel = tk.Frame(content_frame, bg="#34495e", width=250)
		left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
		left_panel.pack_propagate(False)
		
		tk.Label(
			left_panel, 
			text="Controls", 
			font=("Arial", 14, "bold"),
			bg="#34495e",
			fg="white"
		).pack(pady=15)
		
		# Buttons
		btn_style = {
			"font": ("Arial", 11),
			"width": 20,
			"height": 2,
			"relief": tk.RAISED,
			"bd": 3
		}
		
		self.btn_open = tk.Button(
			left_panel, 
			text="📁 Open Image", 
			command=self.open_image,
			bg="#3498db",
			fg="white",
			**btn_style
		)
		self.btn_open.pack(pady=10, padx=15)
		
		self.btn_save = tk.Button(
			left_panel, 
			text="💾 Save Result", 
			command=self.save_result,
			bg="#27ae60",
			fg="white",
			state=tk.DISABLED,
			**btn_style
		)
		self.btn_save.pack(pady=10, padx=15)
		
		self.btn_camera = tk.Button(
			left_panel, 
			text="📷 Start Camera", 
			command=self.toggle_camera,
			bg="#e67e22",
			fg="white",
			**btn_style
		)
		self.btn_camera.pack(pady=10, padx=15)
		
		self.btn_clear = tk.Button(
			left_panel, 
			text="🗑️ Clear", 
			command=self.clear_display,
			bg="#95a5a6",
			fg="white",
			**btn_style
		)
		self.btn_clear.pack(pady=10, padx=15)
		
		# Zoom controls
		zoom_frame = tk.LabelFrame(
			left_panel, 
			text="Zoom Controls", 
			font=("Arial", 11, "bold"),
			bg="#34495e",
			fg="white",
			bd=2
		)
		zoom_frame.pack(pady=10, padx=15, fill=tk.X)
		
		zoom_btn_style = {
			"font": ("Arial", 12, "bold"),
			"width": 5,
			"height": 1,
			"relief": tk.RAISED,
			"bd": 2
		}
		
		zoom_btn_container = tk.Frame(zoom_frame, bg="#34495e")
		zoom_btn_container.pack(pady=10)
		
		tk.Button(
			zoom_btn_container, 
			text="➕", 
			command=self.zoom_in,
			bg="#16a085",
			fg="white",
			**zoom_btn_style
		).grid(row=0, column=0, padx=5)
		
		tk.Button(
			zoom_btn_container, 
			text="➖", 
			command=self.zoom_out,
			bg="#16a085",
			fg="white",
			**zoom_btn_style
		).grid(row=0, column=1, padx=5)
		
		tk.Button(
			zoom_btn_container, 
			text="↺", 
			command=self.reset_zoom,
			bg="#16a085",
			fg="white",
			**zoom_btn_style
		).grid(row=0, column=2, padx=5)
		
		self.zoom_label = tk.Label(
			zoom_frame,
			text="Zoom: 100%",
			font=("Arial", 10),
			bg="#34495e",
			fg="white"
		)
		self.zoom_label.pack(pady=5)
		
		tk.Label(
			zoom_frame,
			text="💡 Scroll or click buttons\nDrag to pan when zoomed",
			font=("Arial", 8),
			bg="#34495e",
			fg="#95a5a6",
			justify=tk.CENTER
		).pack(pady=5)
		
		# Stats frame
		stats_frame = tk.LabelFrame(
			left_panel, 
			text="Detection Stats", 
			font=("Arial", 11, "bold"),
			bg="#34495e",
			fg="white",
			bd=2
		)
		stats_frame.pack(pady=20, padx=15, fill=tk.X)
		
		self.stats_text = tk.Text(
			stats_frame, 
			height=12, 
			width=25,
			font=("Courier", 9),
			bg="#2c3e50",
			fg="white",
			relief=tk.FLAT
		)
		self.stats_text.pack(pady=10, padx=10)
		self.update_stats()
		
		# Right panel - Image display
		right_panel = tk.Frame(content_frame, bg="#34495e")
		right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
		
		tk.Label(
			right_panel, 
			text="Detection View", 
			font=("Arial", 14, "bold"),
			bg="#34495e",
			fg="white"
		).pack(pady=10)
		
		# Image canvas
		self.canvas_frame = tk.Frame(right_panel, bg="#2c3e50", relief=tk.SUNKEN, bd=2)
		self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
		
		self.canvas = tk.Label(
			self.canvas_frame,
			text="No image loaded\n\nClick 'Open Image' to start",
			font=("Arial", 14),
			bg="#2c3e50",
			fg="#95a5a6"
		)
		self.canvas.pack(fill=tk.BOTH, expand=True)
		
		# Bind mouse wheel for zoom
		self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
		self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
		self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
		
		# Bind mouse drag for panning
		self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
		self.canvas.bind("<B1-Motion>", self.on_drag_motion)
		self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
		
		# Status bar
		self.status_var = tk.StringVar()
		self.status_var.set("Ready")
		status_bar = tk.Label(
			root, 
			textvariable=self.status_var, 
			relief=tk.SUNKEN, 
			anchor=tk.W,
			bg="#34495e",
			fg="white",
			font=("Arial", 9)
		)
		status_bar.pack(side=tk.BOTTOM, fill=tk.X)
	
	def open_image(self):
		path = filedialog.askopenfilename(
			title="Select Microscope Image",
			filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif")]
		)
		if not path:
			return
		
		self.status_var.set("Processing image...")
		self.root.update()
		
		img = cv2.imread(path)
		res = self.model.predict(source=img, device=self.device, conf=0.25, verbose=False)[0]
		
		xyxy = res.boxes.xyxy.cpu().numpy()
		cls = res.boxes.cls.cpu().numpy().astype(int)
		conf = res.boxes.conf.cpu().numpy()
		
		boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in xyxy]
		out = draw_detections(img, boxes, cls.tolist(), conf.tolist(), self.class_names)
		
		self.current_image = out
		self.current_detections = {"boxes": boxes, "classes": cls.tolist(), "confidences": conf.tolist()}
		
		# Reset zoom for new image
		self.zoom_level = 1.0
		self.zoom_label.config(text="Zoom: 100%")
		
		self.display_image(out)
		self.update_stats(len(boxes), cls.tolist(), conf.tolist())
		self.btn_save.config(state=tk.NORMAL)
		self.status_var.set(f"Detected {len(boxes)} parasite(s) - {Path(path).name}")
	
	def save_result(self):
		if self.current_image is None:
			return
		
		path = filedialog.asksaveasfilename(
			defaultextension=".jpg",
			filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
		)
		if path:
			cv2.imwrite(path, self.current_image)
			self.status_var.set(f"Saved to {Path(path).name}")
	
	def toggle_camera(self):
		if self.cam_running:
			self.stop_camera()
		else:
			self.start_camera()
	
	def start_camera(self):
		if self.cam_running:
			return
		self.cam_running = True
		self.btn_camera.config(text="⏹️ Stop Camera", bg="#e74c3c")
		self.btn_open.config(state=tk.DISABLED)
		self.status_var.set("Camera running - Press 'Stop Camera' to end")
		self.cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
		self.cam_thread.start()
	
	def stop_camera(self):
		self.cam_running = False
		self.btn_camera.config(text="📷 Start Camera", bg="#e67e22")
		self.btn_open.config(state=tk.NORMAL)
		self.status_var.set("Camera stopped")
	
	def _cam_loop(self):
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			self.cam_running = False
			self.status_var.set("Error: Could not open camera")
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
			
			self.current_image = out
			self.display_image(out)
			self.update_stats(len(boxes), cls.tolist(), conf.tolist())
			
			if len(boxes) > 0:
				self.btn_save.config(state=tk.NORMAL)
		
		cap.release()
	
	def display_image(self, img_bgr):
		# Convert BGR to RGB
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		
		# Force update to get real canvas size
		self.root.update_idletasks()
		
		# Resize to fit canvas
		canvas_width = self.canvas_frame.winfo_width() - 20
		canvas_height = self.canvas_frame.winfo_height() - 20
		
		# Use larger defaults
		if canvas_width < 200:
			canvas_width = 900
		if canvas_height < 200:
			canvas_height = 680
		
		h, w = img_rgb.shape[:2]
		base_scale = min(canvas_width/w, canvas_height/h)
		
		# Apply zoom level
		scale = base_scale * self.zoom_level
		new_w, new_h = int(w*scale), int(h*scale)
		
		img_resized = cv2.resize(img_rgb, (new_w, new_h))
		
		# Apply panning if zoomed in
		if self.zoom_level > 1.0 and (new_w > canvas_width or new_h > canvas_height):
			# Calculate center point with pan offset
			center_x = new_w // 2 - self.pan_x
			center_y = new_h // 2 - self.pan_y
			
			# Calculate crop boundaries
			x1 = max(0, int(center_x - canvas_width // 2))
			y1 = max(0, int(center_y - canvas_height // 2))
			x2 = min(new_w, int(center_x + canvas_width // 2))
			y2 = min(new_h, int(center_y + canvas_height // 2))
			
			# Adjust if we hit boundaries
			if x2 - x1 < canvas_width:
				if x1 == 0:
					x2 = min(new_w, x1 + canvas_width)
				else:
					x1 = max(0, x2 - canvas_width)
			
			if y2 - y1 < canvas_height:
				if y1 == 0:
					y2 = min(new_h, y1 + canvas_height)
				else:
					y1 = max(0, y2 - canvas_height)
			
			# Crop the image
			img_resized = img_resized[y1:y2, x1:x2]
		
		# Convert to PIL and display
		img_pil = Image.fromarray(img_resized)
		img_tk = ImageTk.PhotoImage(img_pil)
		
		self.canvas.config(image=img_tk, text="")
		self.canvas.image = img_tk
	
	def update_stats(self, count=0, classes=None, confidences=None):
		self.stats_text.delete(1.0, tk.END)
		
		stats = f"Total Detected: {count}\n"
		stats += "=" * 25 + "\n\n"
		
		if classes and confidences:
			# Count by class
			class_counts = {}
			class_conf_sum = {}
			for c, conf in zip(classes, confidences):
				class_name = self.class_names[c]
				class_counts[class_name] = class_counts.get(class_name, 0) + 1
				class_conf_sum[class_name] = class_conf_sum.get(class_name, 0) + conf
			
			for class_name in sorted(class_counts.keys()):
				cnt = class_counts[class_name]
				avg_conf = class_conf_sum[class_name] / cnt
				stats += f"{class_name.capitalize()}:\n"
				stats += f"  Count: {cnt}\n"
				stats += f"  Avg Conf: {avg_conf:.2%}\n\n"
		else:
			stats += "No detections yet\n\n"
		
		stats += "=" * 25 + "\n"
		stats += "Classes:\n"
		for i, name in enumerate(self.class_names):
			stats += f"  {name.capitalize()}\n"
		
		self.stats_text.insert(1.0, stats)
	
	def clear_display(self):
		self.current_image = None
		self.current_detections = None
		self.zoom_level = 1.0
		self.pan_x = 0
		self.pan_y = 0
		self.zoom_label.config(text="Zoom: 100%")
		self.canvas.config(
			image="",
			text="No image loaded\n\nClick 'Open Image' to start"
		)
		self.update_stats()
		self.btn_save.config(state=tk.DISABLED)
		self.status_var.set("Display cleared")
	
	def zoom_in(self):
		if self.current_image is None:
			return
		self.zoom_level = min(self.zoom_level * 1.25, 5.0)
		self.zoom_label.config(text=f"Zoom: {int(self.zoom_level*100)}%")
		self.display_image(self.current_image)
	
	def zoom_out(self):
		if self.current_image is None:
			return
		self.zoom_level = max(self.zoom_level / 1.25, 0.1)
		self.zoom_label.config(text=f"Zoom: {int(self.zoom_level*100)}%")
		self.display_image(self.current_image)
	
	def reset_zoom(self):
		if self.current_image is None:
			return
		self.zoom_level = 1.0
		self.pan_x = 0
		self.pan_y = 0
		self.zoom_label.config(text="Zoom: 100%")
		self.display_image(self.current_image)
	
	def on_mouse_wheel(self, event):
		if self.current_image is None:
			return
		
		# Determine zoom direction
		if event.num == 5 or event.delta < 0:
			# Zoom out
			self.zoom_out()
		elif event.num == 4 or event.delta > 0:
			# Zoom in
			self.zoom_in()
	
	def on_drag_start(self, event):
		if self.current_image is None or self.zoom_level <= 1.0:
			return
		self.is_dragging = True
		self.drag_start_x = event.x
		self.drag_start_y = event.y
		self.canvas.config(cursor="fleur")  # Change cursor to move icon
	
	def on_drag_motion(self, event):
		if not self.is_dragging or self.current_image is None:
			return
		
		# Calculate drag delta
		dx = event.x - self.drag_start_x
		dy = event.y - self.drag_start_y
		
		# Update pan position
		self.pan_x += dx
		self.pan_y += dy
		
		# Update drag start for next motion
		self.drag_start_x = event.x
		self.drag_start_y = event.y
		
		# Redraw image with new pan
		self.display_image(self.current_image)
	
	def on_drag_end(self, event):
		self.is_dragging = False
		self.canvas.config(cursor="")


def main():
	parser = argparse.ArgumentParser(description="Advanced GUI for Plasmodium Detector")
	parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--data", type=str, default="config/data.yaml")
	args = parser.parse_args()
	
	class_names = load_class_names_from_data_yaml(args.data)
	root = tk.Tk()
	app = PlasmodiumDetectorGUI(root, args.weights, args.device, class_names)
	root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
	root.mainloop()


if __name__ == "__main__":
	main()

