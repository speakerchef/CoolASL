import math
import time
import logging as log
import cv2
import mediapipe as mp
import os

import torch
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks
from torchvision.transforms import v2
from PIL import Image
from src.model import LandmarkClassifier, resnet18
from src.dataloader import asl_train_path, normalize_landmarks
from configs.cfg_importer import import_cfg


cfg = import_cfg()
MODEL_MODE = cfg['MODEL_MODE']

mp_modelpath = '../models/hand_landmarker.task'

model_path = torch.load(f'../models/{MODEL_MODE}_asl.pth')
asl_model = LandmarkClassifier() if MODEL_MODE == 'lmc' else resnet18
asl_model.load_state_dict(model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")
log.info(f"Model: {MODEL_MODE}")

asl_model.to(device)

# MP setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

transforms = v2.Compose([
    v2.Resize((224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Store latest result from callback
latest_result = None

@torch.no_grad()
def inference(input):
    """
    Runs inference on real-time landmark data from video stream
    :param input: Input to model
    """
    data = input.to(device)
    outputs = asl_model(data)

    _, predicted = torch.max(outputs, 1)

    class_idx = predicted.item()
    # Get class names from training folders (filter out hidden files)
    classes = sorted([d for d in os.listdir(asl_train_path)
                      if os.path.isdir(os.path.join(asl_train_path, d))])

    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)
    label = f"{classes[class_idx]} ({confidence.item():.0%})"
    cv2.putText(frame, label, (x_btm, y_btm - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    draw_landmarks(frame, hand)

def on_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback for async landmark detector."""
    global latest_result
    latest_result = result

def display_model_input(model_in):
    """Utility to debug model input to resnet by displaying transformed input on stream."""
    display = model_in.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display = display * std + mean
    display = display.clamp(0, 1).permute(1, 2, 0).numpy()
    cv2.imshow("Model Input", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

def get_span(x1, x2, y1, y2):
    """
    :return: Euclidean distance between two points
    """
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=mp_modelpath),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=on_result)

# Process block
asl_model.eval()
with HandLandmarker.create_from_options(options) as landmarker:
    video = cv2.VideoCapture(2)
    if not video.isOpened():
        log.error("No video")

    while (True):
        ret, frame = video.read()
        if not ret:
            log.error("No frames")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR TO RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ftime_ms = int(time.time_ns() // 1000000)
        landmarker.detect_async(image=mp_image, timestamp_ms=ftime_ms)

        # Bounding box & Real-time Inference
        if latest_result and latest_result.hand_landmarks:
            for hand in latest_result.hand_landmarks:

                coord_list = []
                if hand:
                    for lm in latest_result.hand_landmarks[0]:
                        coord_list.extend([lm.x, lm.y, lm.z]) # Making tensor

                coord_list = normalize_landmarks(coord_list)

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

                scale =1.1* math.exp(-2*span) # drop scale factor as span increases
                box_size = span * scale

                x_dampener = 0.5 # narrow bbox width

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
                crop_rgb = transforms(crop_rgb)

                # display_model_input(crop_rgb) # Preview model input frame

                # Pre inference data prep
                crop_rgb = crop_rgb.unsqueeze(0)
                lm_input = torch.tensor(coord_list, dtype=torch.float32)
                lm_input = lm_input.unsqueeze(0)

                chosen_input = (lm_input if MODEL_MODE == 'lmc' else crop_rgb)
                inference(chosen_input) # Switch input based on chosen model


        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
