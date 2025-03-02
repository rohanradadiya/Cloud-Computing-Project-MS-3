from ultralytics import YOLO
import cv2
import torch

#loads the pre-trained YOLO model.
model = YOLO("src/yolov8n.pt")  

def detect_pedestrians(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    pedestrians = []
    for r in results:
        for box in r.boxes:
            if r.names[int(box.cls)] == "person":  # Filter for pedestrians
                pedestrians.append({
                    "bounding_box": box.xyxy.tolist(),
                    "confidence": float(box.conf)
                })
    return pedestrians
