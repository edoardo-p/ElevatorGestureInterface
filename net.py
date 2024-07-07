import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class Net(nn.Module):
    def __init__(self, num_gestures: int, num_landmarks: int = 21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_landmarks, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_gestures),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).softmax(dim=0)


class LandmarkDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.landmarks = data[:, :-1]
        self.gestures = data[:, -1]

    def __len__(self) -> int:
        return self.gestures.shape[0]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.landmarks[item]), torch.tensor(self.gestures[item])
