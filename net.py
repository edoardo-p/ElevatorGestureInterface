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
    "Open doors",
    "Close doors",
]


class GestureNet(nn.Module):
    def __init__(self, num_gestures: int, num_landmarks: int = 21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_landmarks * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_gestures),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).softmax(dim=0)

    def infer(self, landmarks: np.ndarray[np.float32], handedness: np.ndarray[np.float32]) -> str:
        landmarks[:, :, 0] = np.abs(handedness.reshape(-1, 1) - landmarks[:, :, 0])
        tensor = torch.tensor(landmarks, dtype=torch.float32).reshape(len(landmarks), -1)
        prediction = self.net(tensor).argmax(dim=1)
        return GESTURE_NAMES[prediction.mode().values.item()]


class LandmarkDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.landmarks = data[:, :-1]
        gestures = data[:, -1]
        self.one_hot_gestures = nn.functional.one_hot(torch.tensor(gestures, dtype=torch.long), num_classes=10)

    def __len__(self) -> int:
        return self.landmarks.shape[0]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.landmarks[item], dtype=torch.float32), self.one_hot_gestures[item].to(torch.float32)
