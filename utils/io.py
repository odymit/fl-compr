import csv
import json
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from .constants import LOG_DIR


def save_results(args, metrics, results):
    """Save the results of the experiment to a file.

    Args:
        args (argparse.Namespace): Arguments used to run the experiment.
        metrics (dict): Dictionary containing the metrics of the experiment.
        results (dict): Dictionary containing the results of the experiment.
    """

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    # Draw the acc and loss curve
    plt.figure()
    plt.plot(metrics["loss"], label="Loss")
    plt.legend()
    prefix = [args.mode, formatted_time]
    name = "_".join(prefix + ["loss.png"])
    plt.savefig(LOG_DIR / name)

    # Create a new figure for accuracy
    plt.figure()
    plt.plot(metrics["acc"], label="Accuracy")
    plt.legend()
    name = "_".join(prefix + ["acc.png"])
    plt.savefig(LOG_DIR / name)

    name = "_".join(prefix + ["results.json"])
    with open(LOG_DIR / name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    name = "_".join(prefix + ["metrics.json"])
    with open(LOG_DIR / name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


def collect_data_partition(args, trainloaders, num_classes):
    """Collection and analysis partition of the data.

    Args:
        trainloader (torch.utils.data.DataLoader): DataLoader for the training set.

    Returns:
        list: List containing the data samples in different classes.
    """
    data_partitions = []
    for client_idx in range(len(trainloaders)):
        data_partition = {idx: 0 for idx in range(num_classes)}
        for _, target in trainloaders[client_idx]:
            for label in target.numpy():
                if label not in data_partition:
                    raise ValueError(f"Label {label} not in the data partition.")
                data_partition[label] += 1
        print(f"Client {client_idx} data partition: {data_partition}")
        data_partitions.append(data_partition)
    # save data partitions to a csv file
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    prefix = [args.mode, formatted_time]
    name = "-".join(prefix + ["data_partitions.csv"])
    keys = data_partitions[0].keys()
    with open(LOG_DIR / name, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_partitions)
    return data_partitions


def draw_data_partition(args, data_partitions):
    """Draw the data partition of the clients.

    Args:
        data_partitions (list of dict): List containing the dict of data samples in different classes.
    """
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    prefix = [args.mode, formatted_time]
    name = "_".join(prefix + ["data_partition.png"])
    print(name)
    df = pd.DataFrame(data_partitions)
    df.plot.barh(stacked=True)
    plt.tight_layout()
    plt.ylabel("clients")
    plt.xlabel("num_sumples")
    plt.savefig(LOG_DIR / name, dpi=400)
