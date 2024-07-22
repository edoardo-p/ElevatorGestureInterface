from collections import deque

import cv2
import mediapipe as mp
import numpy as np

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections


class DataBuffer:
    def __init__(self, sample_size: int):
        self.sample_size = sample_size
        self.landmarks_buffer: deque[np.ndarray] = deque(maxlen=sample_size)
        self.handedness_buffer: deque[str] = deque(maxlen=sample_size)

    @property
    def is_full(self) -> bool:
        return len(self.landmarks_buffer) == self.sample_size

    def add_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        if not result.hand_landmarks or not result.hand_landmarks[0]:
            self._clear()
            return

        self.handedness_buffer.append(result.handedness[0][0].display_name)
        self._update_landmarks_buffer(result.hand_landmarks[0])

    def display_landmarks(self, image: np.ndarray) -> np.ndarray:
        if not self.landmarks_buffer:
            return image

        height, width, *_ = image.shape
        coords = (self.landmarks_buffer[-1][:, :-1] * (width, height)).astype(np.int32)
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, coords[conn.start], coords[conn.end], (255, 255, 255), 2)

        for x, y in coords:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def _update_landmarks_buffer(self, landmarks) -> None:
        landmark_coords = np.empty((21, 3), dtype=np.float32)
        for i, landmark in enumerate(landmarks):
            landmark_coords[i] = landmark.x, landmark.y, landmark.z
        self.landmarks_buffer.append(landmark_coords)

    def _clear(self) -> None:
        self.landmarks_buffer.clear()
        self.handedness_buffer.clear()
