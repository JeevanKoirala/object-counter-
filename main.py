import os
import cv2
import torch
import requests
import subprocess
from pathlib import Path
from tkinter import Tk, filedialog
import shutil

def install_packages():
    try:
        import torch
        import torchvision
        import cv2
        import requests
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'opencv-python', 'requests'])

install_packages()

class ObjectCounter:
    def __init__(self):
        self.model_path = "yolov5m.pt"
        self.clear_cache()
        self.download_model()
        self.load_model(force_reload=True)  

    def clear_cache(self):
        
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master"
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir)

    def download_model(self):
        try:
            if not Path(self.model_path).is_file():
                model_url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt"
                response = requests.get(model_url, stream=True)
                with open(self.model_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
        except Exception as e:
            print(f"Error downloading the model: {e}")
            self.clear_cache()
            raise

    def load_model(self, force_reload=False):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=force_reload)
        except Exception as e:
            print(f"Error loading the model: {e}")
            self.clear_cache()
            raise

    def process_frame(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        return detections, len(detections)

    def annotate_frame(self, frame, detections, count):
        for x1, y1, x2, y2, conf, cls in detections:
            label = f"{self.model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def run_on_image(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        image = cv2.imread(file_path)
        detections, count = self.process_frame(image)
        print("Detected classes:", [self.model.names[int(cls)] for *_, cls in detections])
        output = self.annotate_frame(image, detections, count)
        cv2.imshow("Object Counter - Image", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_on_video(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if not file_path:
            return
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections, count = self.process_frame(frame)
            print("Detected classes:", [self.model.names[int(cls)] for *_, cls in detections])
            output = self.annotate_frame(frame, detections, count)
            cv2.imshow("Object Counter - Video", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_on_camera(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections, count = self.process_frame(frame)
            print("Detected classes:", [self.model.names[int(cls)] for *_, cls in detections])
            output = self.annotate_frame(frame, detections, count)
            cv2.imshow("Object Counter - Live Camera", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

counter = ObjectCounter()
mode = input("Enter mode (image/video/camera): ").strip().lower()
if mode == "image":
    counter.run_on_image()
elif mode == "video":
    counter.run_on_video()
elif mode == "camera":
    counter.run_on_camera()
else:
    print("Invalid mode.")