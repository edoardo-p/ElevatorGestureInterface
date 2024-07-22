import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torcheval.metrics import (
    Metric,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from conf_mat import plot_confusion_matrix
from net import GESTURE_NAMES, GestureNet, LandmarkDataset

DATA_DIR = Path(r".\data")
MODEL_DIR = Path(r".\models")
EXP_DIR = Path(r".\experiments")
SEED = 42
NUM_GESTURES = 8


def split_data(data: np.ndarray, train_split: float) -> tuple[np.ndarray, np.ndarray]:
    train_data, test_data = [], []
    for count in np.unique(data[:, -1]):
        sub_data = data[data[:, -1] == count]
        np.random.shuffle(sub_data)
        split_index = int(sub_data.shape[0] * train_split)
        train_data.append(sub_data[:split_index])
        test_data.append(sub_data[split_index:])

    return np.concatenate(train_data, axis=0), np.concatenate(test_data, axis=0)


def update_metrics(
    model: GestureNet,
    loader: DataLoader,
    device: torch.device,
    metrics: list[Metric[torch.Tensor]],
) -> None:
    for landmarks, gesture in loader:
        landmarks = landmarks.to(device)
        gesture = gesture.to(device).argmax(dim=1)
        predicted = model(landmarks).argmax(dim=1)

        for metric in metrics:
            metric.update(predicted, gesture)


def train_model(
    model: GestureNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    metrics: dict[Metric[torch.Tensor], str],
) -> dict[str, list[float]]:
    model = model.to(device)
    metric_vals = {}
    metric_objs = []
    tqdm_str = ""
    for metric, name in metrics.items():
        metric_vals[name] = []
        metric_objs.append(metric)
        tqdm_str += f"{name}: %.4f "

    for _ in (pbar := tqdm(range(epochs))):
        model.train()

        for landmarks, gesture in train_loader:
            landmarks = landmarks.to(device)
            gesture = gesture.to(device)

            optimizer.zero_grad()
            predicted = model(landmarks)
            loss_fn(predicted, gesture).backward()
            optimizer.step()

        model.eval()
        update_metrics(model, test_loader, device, metric_objs)
        for metric, name in metrics.items():
            val = metric.compute().item()
            metric_vals[name].append(val)
            metric.reset()

        epoch_metrics = [val[-1] for val in metric_vals.values()]
        pbar.set_description(tqdm_str % tuple(epoch_metrics))

    return metric_vals


def test_model(
    model: GestureNet, loader: DataLoader, device: torch.device
) -> np.ndarray:
    conf_mat = MulticlassConfusionMatrix(num_classes=NUM_GESTURES, device=device)
    model.eval()

    for landmarks, gesture in loader:
        landmarks = landmarks.to(device)
        gesture = gesture.to(device).argmax(dim=1)
        predicted = model(landmarks).argmax(dim=1)
        conf_mat.update(predicted, gesture)

    return conf_mat.compute().numpy()


def main(train=True):
    hyperparams = {
        "train_split": 0.8,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 500,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "average": "macro",
        "num_classes": NUM_GESTURES,
        "device": device,
    }
    metrics_dict = {
        MulticlassAccuracy(**kwargs): "Accuracy",
        MulticlassPrecision(**kwargs): "Precision",
        MulticlassF1Score(**kwargs): "F1 Score",
        MulticlassRecall(**kwargs): "Recall",
    }

    for sub_dir in os.listdir(EXP_DIR):
        np.random.seed(SEED)
        data = np.load(EXP_DIR / sub_dir / "hands.npy")
        train_set, test_set = split_data(data, hyperparams["train_split"])
        train_loader = DataLoader(
            LandmarkDataset(train_set),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )
        test_loader = DataLoader(
            LandmarkDataset(test_set),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )

        runs_metrics = []
        cms = []

        for run in range(5):
            print(f"Training {sub_dir} run {run}")
            num_landmarks = 63 if sub_dir.startswith("xyz") else 42
            model = GestureNet(num_gestures=NUM_GESTURES, num_landmarks=num_landmarks)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
            test_metrics = train_model(
                model,
                train_loader,
                test_loader,
                loss_fn,
                optimizer,
                hyperparams["epochs"],
                device,
                metrics_dict,
            )
            runs_metrics.append(test_metrics)
            torch.save(
                model.state_dict(),
                EXP_DIR / sub_dir / f"{model.__class__.__name__}_{run}.pt",
            )

            cm = test_model(model, test_loader, device)
            cms.append(cm)

        np.save(EXP_DIR / sub_dir / "metrics.npy", np.array(runs_metrics))
        np.save(EXP_DIR / sub_dir / "cms.npy", np.array(cms))


if __name__ == "__main__":
    main(True)
