import cv2
import mediapipe as mp
import numpy as np

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


class DataBuffer:
    def __init__(self):
        self.result_buffer = None
        self.handedness = None
        self.reset_buffers()

    def reset_buffers(self):
        self.result_buffer = np.zeros((21, 2), dtype=np.int32)
        self.handedness = ""

    def result_callback(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if not result.hand_landmarks or not result.hand_landmarks[0]:
            self.reset_buffers()
            return

        self.handedness = result.handedness[0][0].display_name
        self.update_landmark_buffer(
            result.hand_landmarks[0], output_image.width, output_image.height
        )

    def update_landmark_buffer(self, landmarks, width: int, height: int) -> None:
        for i, landmark in enumerate(landmarks):
            self.result_buffer[i] = int(landmark.x * width), int(landmark.y * height)

    def display_landmarks(self, image: np.ndarray) -> np.ndarray:
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, self.result_buffer[conn.start], self.result_buffer[conn.end], (255, 255, 255), 2)

        for i, (x, y) in enumerate(self.result_buffer):
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                image, f"{i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
            )

        if self.handedness is not None:
            hand_center = self.result_buffer.mean(axis=0, dtype=np.int32)
            cv2.putText(image, self.handedness, hand_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        return image
