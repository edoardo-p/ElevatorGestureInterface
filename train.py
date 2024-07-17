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
SEED = 42
NUM_GESTURES = 10


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
        gesture = gesture.to(device)
        predicted = model(landmarks)

        for metric in metrics:
            metric.update(predicted.argmax(dim=1), gesture.argmax(dim=1))


def train(
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

    torch.save(model.state_dict(), MODEL_DIR / f"{model.__class__.__name__}.pt")
    return metric_vals


def test(model: GestureNet, loader: DataLoader, device: torch.device) -> torch.Tensor:
    conf_mat = MulticlassConfusionMatrix(num_classes=NUM_GESTURES, device=device)
    model.load_state_dict(torch.load(MODEL_DIR / f"{model.__class__.__name__}.pt"))
    model.eval()

    for landmarks, gesture in loader:
        landmarks = landmarks.to(device)
        gesture = gesture.to(device)
        predicted = model(landmarks)
        conf_mat.update(predicted.argmax(dim=1), gesture.argmax(dim=1))

    return conf_mat.compute()


def main():
    np.random.seed(SEED)
    data = np.load(DATA_DIR / "hands.npy")
    hyperparams = {
        "train_split": 0.8,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 500,
    }

    train_set, test_set = split_data(data, hyperparams["train_split"])
    train_loader = DataLoader(
        LandmarkDataset(train_set), batch_size=hyperparams["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        LandmarkDataset(test_set), batch_size=hyperparams["batch_size"], shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureNet(num_gestures=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    metrics_dict = {
        MulticlassAccuracy(device=device): "Accuracy",
        MulticlassPrecision(device=device): "Precision",
        MulticlassF1Score(device=device): "F1 Score",
        MulticlassRecall(device=device): "Recall",
    }

    test_metrics = train(
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        hyperparams["epochs"],
        device,
        metrics_dict,
    )
    # print(test_metrics)

    # cm = test(model, test_loader, device)
    # plot_confusion_matrix(cm.to(torch.int64).numpy(), GESTURE_NAMES)


if __name__ == "__main__":
    main()
