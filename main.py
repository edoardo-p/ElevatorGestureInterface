import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

landmark_buffer = np.zeros((21, 2), dtype=np.int32)


def print_result(
        result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    if not result.hand_landmarks or not result.hand_landmarks[0]:
        return

    update_landmark_buffer(
        result.hand_landmarks[0], output_image.width, output_image.height
    )


def take_picture(path: str) -> None:
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Hand recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            cv2.imwrite(path, frame)
            break

    video.release()
    cv2.destroyAllWindows()


def update_landmark_buffer(landmarks, width: int, height: int) -> None:
    for i, landmark in enumerate(landmarks):
        landmark_buffer[i] = int(landmark.x * width), int(landmark.y * height)


def display_landmarks(image: np.ndarray) -> None:
    connections = [
        (0, 1),
        (0, 17),
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (5, 9),
        (6, 7),
        (7, 8),
        (9, 10),
        (9, 13),
        (10, 11),
        (11, 12),
        (13, 14),
        (13, 17),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    for i_start, i_end in connections:
        cv2.line(image, landmark_buffer[i_start], landmark_buffer[i_end], (255, 255, 255), 2)

    for i, (x, y) in enumerate(landmark_buffer):
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
        )


def main():
    data_dir = Path(r".\data")
    models_dir = Path(r".\models")
    model_path = models_dir / "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
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
            display_landmarks(frame)
            cv2.imshow("Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
