import os
import uuid
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
GESTURES = ("0", "1", "2", "3", "4", "5", "u", "d")


def take_picture(path: Path) -> None:
    for gesture in GESTURES:
        os.makedirs(path / gesture, exist_ok=True)

    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Hand recognizer", frame)
        key = chr(cv2.waitKey(1) & 0xFF)

        if key == " ":
            break
        elif key in GESTURES:
            print(key, uuid.uuid4())
            cv2.imwrite(str(path / key / f"{uuid.uuid4()}.png"), frame)

    video.release()
    cv2.destroyAllWindows()


def convert_image_to_numpy(data_dir: Path, model_path: Path) -> None:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    coordinates = []
    gestures = []
    invalid_photos = []

    with HandLandmarker.create_from_options(options) as recognizer:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file == "hands.npy":
                    continue
                image = cv2.imread(rf"{root}\{file}")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                result = recognizer.detect(mp_image)

                if not result.hand_landmarks:
                    print(rf"No hand recognized: {root}\{file}")
                    invalid_photos.append(rf"{root}\{file}")
                    continue

                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].display_name

                if handedness == "Left":
                    # Flips the x-coordinates to always have a 'right' hand
                    coords = [(1 - landmark.x, landmark.y) for landmark in landmarks]
                elif handedness == "Right":
                    coords = [(landmark.x, landmark.y) for landmark in landmarks]
                else:
                    raise ValueError(f"Invalid handedness: {handedness}")

                coordinates.append(np.array(coords).flatten())
                gestures.append([GESTURES.index(root.split("\\")[-1])])

        np.save(data_dir / "hands.npy", np.hstack((coordinates, gestures)))
        for photo in invalid_photos:
            os.remove(photo)
        print(f"Removed {len(invalid_photos)} photos of unrecognized hands")


def print_data_description(path: Path) -> None:
    arr = np.load(path)
    print(f"{arr.shape}, {arr.dtype}")
    print(f"Unique gestures: {np.unique(arr[:, -1], return_counts=True)}")


if __name__ == "__main__":
    # take_picture(DATA_DIR / "images")
    convert_image_to_numpy(DATA_DIR, MODELS_DIR / "hand_landmarker.task")
    print_data_description(DATA_DIR / "hands.npy")
