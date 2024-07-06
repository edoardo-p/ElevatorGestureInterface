import time
from pathlib import Path

import cv2
import mediapipe as mp

from vis import DataBuffer

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DATA_DIR = Path(r".\data")
MODELS_DIR = Path(r".\models")

MAX_HANDS = 1


def main():
    model_path = MODELS_DIR / "hand_landmarker.task"
    buffer = DataBuffer()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=MAX_HANDS,
        result_callback=buffer.result_callback,
    )

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
            cv2.imshow("Landmarks", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
