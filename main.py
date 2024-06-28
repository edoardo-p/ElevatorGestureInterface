from pathlib import Path

import cv2
import mediapipe as mp

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


def main():
    models_dir = Path(r".\models")
    model_path = models_dir / "gesture_recognizer.task"

    video = cv2.VideoCapture(0)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
    )
    with GestureRecognizer.create_from_options(options) as recognizer:
        # The detector is initialized. Use it here.
        # ...
        while True:
            _, frame = video.read()

            cv2.imshow("Face Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

            mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=frame)
            recognizer.recognize_async(mp_image, recognizer.get_timestamp_ms())

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
