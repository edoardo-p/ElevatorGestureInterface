from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(
    result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int
):
    print("gesture recognition result: {}".format(result))


def take_picture(path: str) -> None:
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            cv2.imwrite(path, frame)
            break

    video.release()
    cv2.destroyAllWindows()


def convert_landmarks_to_coordinates(
    landmarks, width: int, height: int
) -> list[tuple[int, int]]:
    return [
        (int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks
    ]


def display_landmarks(image: np.ndarray, coords: list[tuple[int, int]]) -> None:
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
        cv2.line(image, coords[i_start], coords[i_end], (255, 255, 255), 2)

    for i, (x, y) in enumerate(coords):
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
        )

    while True:
        cv2.imshow("Landmarks", image)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    cv2.destroyAllWindows()


def main():
    data_dir = Path(r".\data")
    models_dir = Path(r".\models")
    model_path = models_dir / "gesture_recognizer.task"

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        # result_callback=print_result,
    )

    # take_picture((data_dir / "hand.png").as_posix())
    with GestureRecognizer.create_from_options(options) as recognizer:
        hand = cv2.imread((data_dir / "hand.png").as_posix())
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand)
        data = recognizer.recognize(mp_image)
        coords = convert_landmarks_to_coordinates(
            data.hand_landmarks[0], hand.shape[1], hand.shape[0]
        )
        display_landmarks(hand, coords)


if __name__ == "__main__":
    main()
