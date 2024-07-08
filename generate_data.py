import os
import uuid
from itertools import chain
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DATA_DIR = Path(r".\data")
MODELS_DIR = Path(r".\models")
GESTURES = ("0", "1", "2", "3", "4", "5", "u", "d", "o", "c")


def take_picture(path: Path) -> None:
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Hand recognizer", frame)
        key = chr(cv2.waitKey(1) & 0xFF)

        if key == " ":
            break
        elif key in GESTURES:
            cv2.imwrite(str(path / key / str(uuid.uuid4())), frame)

    video.release()
    cv2.destroyAllWindows()


def convert_image_to_numpy(data_dir: Path, model_path: Path) -> None:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    coordinates = []

    with HandLandmarker.create_from_options(options) as recognizer:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                image = cv2.imread(rf"{root}\{file}")
                width, height = image.shape
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                result = recognizer.detect(mp_image)

                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].display_name

                if handedness == "Left":
                    # Flips the x-coordinates to always have a 'right' hand
                    coords = [(int((1 - landmark.x) * width), int(landmark.y * height)) for landmark in landmarks]
                elif handedness == "Right":
                    coords = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
                else:
                    raise ValueError(f"Invalid handedness: {handedness}")

                flattened_coords = list(chain(*coords)).append(GESTURES.index(root.split(r"/")[-1]))
                coordinates.append(flattened_coords)

        np.save(data_dir / "hands.npy", np.array(coordinates))


if __name__ == '__main__':
    take_picture(DATA_DIR)
    convert_image_to_numpy(DATA_DIR, MODELS_DIR / "hand_landmarker.task")
