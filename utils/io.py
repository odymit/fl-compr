import csv
import json
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from .constants import LOG_DIR


def draw_curve(data, label, filepath):
    """Draw the curve of the data.

    Args:
        data (list): List containing the data.
        label (str): Label of the curve.
        filepath (str): Filepath to save the curve.
    """
    plt.figure()
    plt.plot(data, label=label)
    plt.legend()
    plt.savefig(filepath)


def save_results(args, metrics, results):
    """Save the results of the experiment to a file.

    Args:
        args (argparse.Namespace): Arguments used to run the experiment.
        metrics (dict): Dictionary containing the metrics of the experiment.
        results (dict): Dictionary containing the results of the experiment.
    """

    # Draw the acc and loss curve
    file_save_path = get_save_path(args, "loss", ".png")
    draw_curve(metrics["loss"], "loss", file_save_path)

    # Create a new figure for accuracy
    file_save_path = get_save_path(args, "acc", ".png")
    draw_curve(metrics["acc"], "acc", file_save_path)

    file_save_path = get_save_path(args, "results", ".json")
    with open(file_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    file_save_path = get_save_path(args, "metrics", ".json")
    with open(file_save_path, "w", encoding="utf-8") as f:
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
    save_file_path = get_save_path(args, "data_partitions", ".csv")
    keys = data_partitions[0].keys()
    with open(save_file_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data_partitions)
    return data_partitions


def draw_data_partition(args, data_partitions):
    """Draw the data partition of the clients.

    Args:
        data_partitions (list of dict): List containing the dict of data samples in different classes.
    """
    file_save_path = get_save_path(args, "data_partitions", ".png")
    df = pd.DataFrame(data_partitions)
    df.plot.barh(stacked=True)
    plt.tight_layout()
    plt.ylabel("clients")
    plt.xlabel("num_sumples")
    plt.savefig(file_save_path, dpi=400)


def get_task_prefix(args):
    """Get the prefix of the task.

    Args:
        args (argparse.Namespace): Arguments used to run the experiment.

    Returns:
        tuple: Tuple containing the prefix and the formatted time.
    """
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    prefix = [
        args.mode,
        str(args.rounds),
        str(args.num_clients),
        str(args.learning_rate),
    ]
    return prefix, formatted_time


def get_save_path(args, filename, suffix, with_time=True):
    """Get the save path of the file.

    Args:
        args (argparse.Namespace): Arguments used to run the experiment.
        filename (str): Filename of the file.
        suffix (str): Suffix of the file.

    Returns:
        str: The save path of the file.
    """
    prefix, formatted_time = get_task_prefix(args)
    if with_time:
        name = "_".join(prefix + [filename, formatted_time + suffix])
    else:
        name = "_".join(prefix + [filename + suffix])
    return LOG_DIR / name
