import torch
import cv2
import numpy as np

# Load pre-trained depth model
depth_model = torch.load("models/depth_model.pth")

def estimate_depth(image_path, bounding_boxes):
    img = cv2.imread(image_path)

    # Convert to tensor and pass through model
    input_tensor = torch.tensor(img).float().unsqueeze(0)
    depth_map = depth_model(input_tensor).squeeze(0).detach().numpy()

    depths = []
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box["bounding_box"][0]) 
        pedestrian_depth = np.mean(depth_map[y1:y2, x1:x2])
        depths.append(pedestrian_depth)

    return depths
