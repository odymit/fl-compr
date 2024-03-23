from datetime import datetime
import json
from matplotlib import pyplot as plt
import numpy as np

from .constants import DATA_DIR


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
    # Create a new figure for loss
    plt.figure()
    plt.plot(metrics["loss"], label="Loss")
    plt.legend()
    prefix = [args.mode, formatted_time]
    name = "_".join(prefix + ["loss.png"])
    plt.savefig(f"./logs/{name}")

    # Create a new figure for accuracy
    plt.figure()
    plt.plot(metrics["acc"], label="Accuracy")
    plt.legend()
    name = "_".join(prefix + ["acc.png"])
    plt.savefig(f"./logs/{name}")

    name = "_".join(prefix + ["results.json"])
    with open(f"./logs/{name}", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    name = "_".join(prefix + ["metrics.json"])
    with open(f"./logs/{name}", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
