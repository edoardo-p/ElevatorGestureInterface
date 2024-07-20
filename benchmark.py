import time
from pathlib import Path

import torch

from net import GestureNet

MODELS_DIR = Path(r".\models")


def main():
    model = GestureNet(num_gestures=8)
    model.load_state_dict(torch.load(MODELS_DIR / "GestureNet.pt"))

    tensors = (
        torch.randn(1, 42),
        torch.randn(20, 42),
        torch.randn(50, 42),
        torch.randn(100, 42),
    )
    for tensor in tensors:
        start = time.perf_counter_ns()
        model(tensor)
        end = time.perf_counter_ns()
        print(f"Time: {(end - start)} ns")


if __name__ == "__main__":
    main()
