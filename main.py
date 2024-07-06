import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode

DATA_DIR = Path(r".\data")
MODELS_DIR = Path(r".\models")

MAX_HANDS = 1

landmark_buffer = np.zeros((21, 2), dtype=np.int32)
handedness = ""


def print_result(
        result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global landmark_buffer, handedness
    if not result.hand_landmarks or not result.hand_landmarks[0]:
        landmark_buffer = np.zeros((21, 2), dtype=np.int32)
        handedness = ""
        return

    handedness = result.handedness[0][0].display_name
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
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(image, landmark_buffer[conn.start], landmark_buffer[conn.end], (255, 255, 255), 2)

    for i, (x, y) in enumerate(landmark_buffer):
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
        )

    if handedness is not None:
        hand_center = landmark_buffer.mean(axis=0, dtype=np.int32)
        cv2.putText(image, handedness, hand_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))


def main():
    model_path = MODELS_DIR / "hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=MAX_HANDS,
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
