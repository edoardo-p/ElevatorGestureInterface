import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from buffer import DataBuffer
from net import GestureNet

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODELS_DIR = Path(r".\models")

MAX_HANDS = 1


def main():
    buffer = DataBuffer(sample_size=20)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELS_DIR / "hand_landmarker.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=MAX_HANDS,
        result_callback=buffer.add_result,
    )

    model = GestureNet(num_gestures=8)
    model.load_state_dict(torch.load(MODELS_DIR / "GestureNet.pt"))

    video = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as recognizer:
        start_time = time.time()
        while True:
            ret, frame = video.read()
            timestamp = (time.time() - start_time) * 1000
            if not ret:
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            recognizer.detect_async(mp_image, int(timestamp))
            annotated_frame = buffer.display_landmarks(frame)
            if buffer.is_full:
                gesture = model.infer(
                    np.array(buffer.landmarks_buffer),
                    np.array(
                        [
                            1 if hand == "Left" else 0
                            for hand in buffer.handedness_buffer
                        ]
                    ),
                )
                cv2.putText(
                    annotated_frame,
                    gesture,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                )
            cv2.imshow("Landmarks", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
