import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

GESTURE_NAMES = [
    "Ground floor",
    "First floor",
    "Second floor",
    "Third floor",
    "Fourth floor",
    "Fifth floor",
    "Up",
    "Down",
]


class GestureNet(nn.Module):
    def __init__(self, num_gestures: int, num_landmarks: int = 42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_landmarks, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, num_gestures),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).softmax(dim=0)

    def infer(
        self, landmarks: np.ndarray, handedness: np.ndarray, confidence: float
    ) -> str:
        landmarks[:, :, 0] = np.abs(handedness.reshape(-1, 1) - landmarks[:, :, 0])
        landmarks -= landmarks.mean(axis=1, keepdims=True)
        landmarks = landmarks.reshape(landmarks.shape[0], -1)
        tensor = torch.tensor(landmarks, dtype=torch.float32)
        prediction = self.net(tensor).argmax(dim=1)
        top = prediction.mode().values.item()
        if (prediction == top).sum() / prediction.size(0) < confidence:
            return ""
        return GESTURE_NAMES[prediction.mode().values.item()]


class LandmarkDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.landmarks = data[:, :-1]
        gestures = data[:, -1]
        self.one_hot_gestures = nn.functional.one_hot(
            torch.tensor(gestures, dtype=torch.long)
        )

    def __len__(self) -> int:
        return self.landmarks.shape[0]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(
            self.landmarks[item], dtype=torch.float32
        ), self.one_hot_gestures[item].to(torch.float32)
