import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_pic(path):
    path = Path(path)
    # Read the CSV file
    data = pd.read_csv(path, header=None)
    filename = path.stem
    root = path.parent
    acc = data[1]
    loss = data[2]

    # Plot the accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label="Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = root / f"{filename}_acc.png"
    plt.savefig(acc_path)

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label="Loss", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = root / f"{filename}_loss.png"
    plt.savefig(loss_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    draw_pic(args.path)
