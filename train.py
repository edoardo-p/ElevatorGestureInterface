from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from net import Net, LandmarkDataset

DATA_DIR = Path(r".\data")
SEED = 42


def split_data(data: np.ndarray, train_split: float) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(data.shape[0] * train_split)
    np.random.shuffle(data)
    return data[:split_index], data[split_index:]


def train(model: torch.nn.Module, loader: DataLoader, loss_fn):
    model.train()
    for landmarks, gesture in loader:
        predict = model(landmarks)
        loss_fn(predict).backwards()

        # Metrics


def main():
    np.random.seed(SEED)
    data = np.load(DATA_DIR / "hands.npy")

    train_set, test_set = split_data(data, 0.8)
    train_loader = DataLoader(LandmarkDataset(train_set), batch_size=16, shuffle=True)
    test_loader = DataLoader(LandmarkDataset(test_set), batch_size=16, shuffle=True)

    model = Net(num_gestures=5)
    train(model, train_loader)


if __name__ == '__main__':
    main()
