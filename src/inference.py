import math
import time

import cv2
import mediapipe as mp
import numpy as np
import os
from configs.cfg_importer import import_cfg

cfg = import_cfg()

mp_modelpath = os.path.join(cfg['working_path'], 'models/hand_landmarker.task')

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

                # get max span between 5 fingertips and
                # wrist to define bounding box area dynamically
                span = max(get_span(hand[4].x, wrist.x, hand[4].y, wrist.y),
                get_span(hand[8].x, wrist.x, hand[8].y, wrist.y),
                get_span(hand[12].x, wrist.x, hand[12].y, wrist.y),
                get_span(hand[16].x, wrist.x, hand[16].y, wrist.y),
                get_span(hand[20].x, wrist.x, hand[20].y, wrist.y),
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

                # draw landmarks
                #     x = int(landmark.x * w)
                #     y = int(landmark.y * h)
                #     # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                #     x_t1 = landmark.x + 112
                #     y_t1 = landmark.y + 112

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()



