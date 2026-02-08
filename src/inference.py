import math
import time

import cv2
import mediapipe as mp
import numpy as np
import os

import torch
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from src.model import model
from src.dataloader import asl_train
from configs.cfg_importer import import_cfg

cfg = import_cfg()

mp_modelpath = os.path.join(cfg['working_path'], 'models/hand_landmarker.task')

asl_path = torch.load('../models/resnet18_asl_08_02_2026_00_22_59')
asl_model = model
asl_model.load_state_dict(asl_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

asl_model.to(device)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Store latest result from callback
latest_result = None

def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def get_span(x1, x2, y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mp_modelpath),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=on_result)


asl_model.eval()
with HandLandmarker.create_from_options(options) as landmarker:
    video = cv2.VideoCapture(2)
    if not video.isOpened():
        print("No video")

    while (True):
        ret, frame = video.read()
        if not ret:
            print("No frames")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR TO RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ftime_ms = int(time.time_ns() // 1000000)
        landmarker.detect_async(image=mp_image, timestamp_ms=ftime_ms)

        # Bounding box creation
        if latest_result and latest_result.hand_landmarks:
            for hand in latest_result.hand_landmarks:
                h, w, _ = frame.shape
                center = hand[9] # middle of hand
                cx = center.x
                cy = center.y
                wrist = hand[0] # wrist landmark

                span = 0

                # get max span between 5 fingertips and
                # wrist to define bounding box area dynamically
                span_y = max(wrist.y, hand[4].y) # Best y b/w thumb and wrist
                span = max(get_span(hand[4].x, wrist.x, hand[4].y, span_y),
                get_span(hand[8].x, wrist.x, hand[8].y, span_y),
                get_span(hand[12].x, wrist.x, hand[12].y, span_y),
                get_span(hand[16].x, wrist.x, hand[16].y, span_y),
                get_span(hand[20].x, wrist.x, hand[20].y, span_y),
                    )

                scale = math.exp(-span) # drop scale factor as span increases
                box_size = span * scale

                x_dampener = 0.7 # narrow bbox width

                x_top = int((cx + box_size * x_dampener) * w)
                y_top = int((cy + box_size) * h)
                x_btm = int((cx - box_size * x_dampener) * w)
                y_btm = int((cy - box_size) * h)

                # Clamp to frame boundaries
                x_btm = max(0, x_btm)
                y_btm = max(0, y_btm)
                x_top = min(w, x_top)
                y_top = min(h, y_top)

                # draw bbox
                cv2.rectangle(frame, (x_top,y_top), (x_btm, y_btm), (0,255,0), 2)

                crop = frame[y_btm:y_top, x_btm:x_top]
                crop_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                transforms = v2.Compose([
                    v2.Resize((224,224), antialias=True),
                    v2.ToImage(),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                crop_rgb = transforms(crop_rgb)

                with torch.no_grad():
                    crop_rgb = crop_rgb.unsqueeze(0)
                    crop_rgb = crop_rgb.to(device)
                    outputs = asl_model(crop_rgb)
                    _, predicted = torch.max(outputs, 1)
                    class_idx = predicted.item()
                    classes = sorted(os.listdir(asl_train))  # ['0', '1', ..., '9', 'A', 'B', ..., 'Z']
                    classes = classes[1:]
                    label = classes[class_idx]
                    print(label)


        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



