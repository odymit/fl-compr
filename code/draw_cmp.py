import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_pic(args):
    path1 = Path(args.path1)
    # Read the CSV file
    data1 = pd.read_csv(path1, header=None)
    filename1 = path1.stem
    root = path1.parent
    acc1 = data1[1]
    loss1 = data1[2]
    path2 = Path(args.path2)
    # Read the CSV file
    data2 = pd.read_csv(path2, header=None)
    filename2 = path2.stem
    acc2 = data2[1]
    loss2 = data2[2]
    metric = [loss1, loss2] if args.loss else [acc1, acc2]
    label = "Loss" if args.loss else "Accuracy"

    # Plot the accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(metric[0], label=filename1, color="red")
    plt.plot(metric[1], label=filename2, color="blue")
    plt.title(f"{label} Curve")
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.legend()
    acc_path = root / f"compr_{filename1}_{filename2}.png"
    plt.savefig(acc_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str)
    parser.add_argument("--path2", type=str)
    parser.add_argument("--loss", default=False, action="store_true")

    args = parser.parse_args()
    draw_pic(args)
