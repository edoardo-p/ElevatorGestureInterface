from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from net import GestureNet, LandmarkDataset

DATA_DIR = Path(r".\data")
MODEL_DIR = Path(r".\models")
SEED = 42


def split_data(data: np.ndarray, train_split: float) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(data.shape[0] * train_split)
    np.random.shuffle(data)
    return data[:split_index], data[split_index:]


def calc_mse(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    mses = []
    for landmarks, gesture in loader:
        landmarks = landmarks.to(device)
        gesture = gesture.to(device)

        predicted = model(landmarks).argmax(dim=1)

        # Calc MSE
        mse = torch.square(gesture - predicted)
        mses.append(mse)

    return np.array(mses).flatten().mean()


def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: torch.nn.modules.loss,
        optimizer: torch.optim,
        epochs: int,
        device: torch.device,
) -> list[float]:
    model = model.to(device)
    best_mse = float("inf")
    mses = []

    for epoch in tqdm(range(epochs)):
        model.train()

        for landmarks, gesture in train_loader:
            landmarks = landmarks.to(device)
            gesture = gesture.to(device)

            optimizer.zero_grad()
            predicted = model(landmarks)
            loss_fn(predicted, gesture).backward()
            optimizer.step()

        # Eval metrics
        model.eval()
        mse = calc_mse(model, test_loader, device)
        mses.append(mse)
        print(f"Epoch {epoch:03d}\t MSE: {sum(mses) / len(mses):.4f}")

        if mse < best_mse:
            torch.save(model, f"{MODEL_DIR}{model.__class__.__name__}.pt")

    return mses


def main():
    np.random.seed(SEED)
    data = np.load(DATA_DIR / "hands.npy")
    hyperparams = {
        "train_split": 0.8,
        "lr": 0.001,
        "batch_size": 16,
        "epochs": 50,
    }

    train_set, test_set = split_data(data, hyperparams["train_split"])
    train_loader = DataLoader(LandmarkDataset(train_set), batch_size=hyperparams["batch_size"], shuffle=True)
    test_loader = DataLoader(LandmarkDataset(test_set), batch_size=hyperparams["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureNet(num_gestures=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    mses = train(model, train_loader, test_loader, loss_fn, optimizer, hyperparams["epochs"], device)
    print(mses)


if __name__ == '__main__':
    main()
