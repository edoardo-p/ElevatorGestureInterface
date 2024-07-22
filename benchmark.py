import os
import time
from pathlib import Path

import torch

from net import GestureNet

MODELS_DIR = Path(r".\experiments")


def main():
    for sub_dir in os.listdir(MODELS_DIR):
        print(f"Benchmarking {sub_dir}")
        num_landmarks = 63 if sub_dir.startswith("xyz") else 42
        model = GestureNet(num_gestures=8, num_landmarks=num_landmarks)
        model.load_state_dict(torch.load(MODELS_DIR / sub_dir / "GestureNet_0.pt"))

        tensors = (
            torch.randn(1, num_landmarks),
            torch.randn(20, num_landmarks),
            torch.randn(50, num_landmarks),
            torch.randn(100, num_landmarks),
        )
        for tensor in tensors:
            start = time.perf_counter_ns()
            model(tensor)
            end = time.perf_counter_ns()
            print(f"Time: {(end - start)} ns")


if __name__ == "__main__":
    main()
