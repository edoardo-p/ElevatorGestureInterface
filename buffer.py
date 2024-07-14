from collections import deque

import cv2
import mediapipe as mp
import numpy as np

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


class DataBuffer:
    def __init__(self, sample_size: int):
        self.landmarks_buffer: deque[np.ndarray] = deque(maxlen=sample_size)
        self.handedness_buffer: deque[str] = deque(maxlen=sample_size)

    def clear(self):
        self.landmarks_buffer.clear()
        self.handedness_buffer.clear()

    def add_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        if not result.hand_landmarks or not result.hand_landmarks[0]:
            self.clear()
            return

        self.handedness_buffer.append(result.handedness[0][0].display_name)
        self._update_landmarks_buffer(
            result.hand_landmarks[0]
        )

    def _update_landmarks_buffer(self, landmarks) -> None:
        landmark_coords = np.empty((21, 2), dtype=np.float32)
        for i, landmark in enumerate(landmarks):
            landmark_coords[i] = landmark.x, landmark.y
        self.landmarks_buffer.append(landmark_coords)

    def display_landmarks(self, image: np.ndarray) -> np.ndarray:
        if not self.landmarks_buffer:
            return image

        height, width, *_ = image.shape
        coords = (self.landmarks_buffer[-1] * (width, height)).astype(np.int32)
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, coords[conn.start], coords[conn.end], (255, 255, 255), 2)

        for i, (x, y) in enumerate(coords):
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        hand_center = coords.mean(axis=0, dtype=np.int32)
        cv2.putText(image, self.handedness_buffer[-1], hand_center, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        return image
