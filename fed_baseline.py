import argparse
import os
import time
from collections import OrderedDict
from datetime import datetime
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common.logger import log

from utils.constants import LOG_DIR

# disable ray duplicate logs
os.environ["RAY_DEDUP_LOGS"] = "0"

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from utils.eval import calc_data
from utils.io import (
    collect_data_partition,
    draw_curve,
    draw_data_partition,
    get_save_path,
    get_task_prefix,
    save_results,
)

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################


# default experiments settings
NUM_CLIENTS = 2
BATCH_SIZE = 128
MOMENTUM = 0  # 0.9
LEARNIGN_RATE = 5e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 400
SEED = 42
# REPEATS = 3
# WARMUP = 5

# set client device using gpu or cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    INFO,
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}",
)


def get_model():
    class Net(nn.Module):
        """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    # net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=10).to(DEVICE)
    net = Net()
    return net


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, criterion, optimizer, scheduler=None):
    """Train the network on the training set."""
    net.train()
    costs = []
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in tqdm(trainloader):
            start = time.time()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            end = time.time()
            cost = end - start
            costs.append(cost)
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        log(
            INFO,
            f"Epoch {epoch+1}: train loss {epoch_loss}, train accuracy {epoch_acc}",
        )
        # log(INFO, "Learning Rate after step:", scheduler.get_last_lr())

    # return time cost per batch
    return sum(costs) / len(costs)


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(SEED))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(SEED)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


# Define Flower client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)


class FlowerClient(fl.client.Client):
    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        log_name,
        uncompr_fn=parameters_to_ndarrays,
        compr_fn=ndarrays_to_parameters,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.uncompr_fn = uncompr_fn
        self.compr_fn = compr_fn
        self.log_name = log_name
        # init training parameters

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=LEARNIGN_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer=self.optimizer, start_factor=0.1, total_iters=WARMUP
        # )
        # train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=self.optimizer, milestones=[150, 250], gamma=0.1
        # )
        # self.scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer=self.optimizer,
        #     schedulers=[warmup_scheduler, train_scheduler],
        #     milestones=[WARMUP],
        # )
        self._get_logger()

    def _get_logger(self):
        fl.common.logger.configure(
            identifier="BaselineExperiment", filename=LOG_DIR / self.log_name
        )

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"[Client {self.cid}] fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_received = ins.parameters
        ndarrays_received = self.uncompr_fn(parameters_received)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_received)

        time_cost_avg = train(
            self.net,
            self.trainloader,
            epochs=1,
            criterion=self.criterion,
            optimizer=self.optimizer,
            # scheduler=self.scheduler,
        )
        # log(INFO, f"[Client {self.cid}] - Learning rate: {self.scheduler.get_last_lr()}")
        ndarrays_updated = get_parameters(self.net)
        # np.savez(f"client_{self.cid}_parameters.npz", *ndarrays_updated)
        # log(
        #     INFO,
        #     f"-------------------------------data saved for client {self.cid}-----------------------------"
        # )
        bytes = calc_data(ndarrays_updated)
        log(INFO, f"[Client {self.cid}] ndarrays byte size: {bytes}")
        # Serialize ndarray's into a Parameters object using our custom function
        if self.compr_fn == ndarrays_to_parameters:
            parameters_updated = self.compr_fn(ndarrays_updated)
        else:
            parameters_updated, bytes = self.compr_fn(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={"time": time_cost_avg, "bytes": bytes},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_received = ins.parameters
        ndarrays_received = self.uncompr_fn(parameters_received)

        set_parameters(self.net, ndarrays_received)
        loss, accuracy = test(self.net, self.valloader)
        log(
            INFO,
            "[Client {}] - Evaluation loss: {}, accuracy on valid set: {}".format(
                self.cid, loss, accuracy
            ),
        )

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


# define the server side
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedBaseline(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        uncompr_fn=parameters_to_ndarrays,
        compr_fn=ndarrays_to_parameters,
    ) -> None:
        """Custom FedAvg strategy with sparse matrices.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.compr_fn = compr_fn
        self.uncompr_fn = uncompr_fn

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        # We deserialize using our custom method
        ndarrays = self.uncompr_fn(parameters)

        eval_res = self.evaluate_fn(server_round, ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # We deserialize each of the results with our custom method
        weights_results = [
            (self.uncompr_fn(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        log(
            INFO,
            f"-------------------------------weights_results-----------------------------",
        )

        # We serialize the aggregated result using our cutom method
        parameters_aggregated = self.compr_fn(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        # Save data and time cost
        bytes_lst = [res.metrics["bytes"] for _, res in results]
        time_cost = [res.metrics["time"] for _, res in results]
        final_results["data"] = bytes_lst
        final_results["time"] = time_cost
        self.parameters_aggregated = parameters_aggregated
        return parameters_aggregated, metrics_aggregated


def get_evaluate_fn(net, testloader):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_parameters(net, parameters)
        net.to(DEVICE)
        total_loss, total_correct, total_samples = 0, 0, 0
        # Do not calculate gradients
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                # Forward pass
                output = net(data)
                # Calculate loss
                loss = F.cross_entropy(output, target, reduction="sum")
                # Get predictions
                pred = output.argmax(dim=1)
                # Calculate correct predictions
                correct = pred.eq(target.view_as(pred)).sum().item()

                total_loss += loss.item()
                total_correct += correct
                total_samples += data.shape[0]

        loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        global metrics
        metrics["round"].append(server_round)
        metrics["acc"].append(accuracy)
        metrics["loss"].append(loss)
        final_results["acc"].append(accuracy)
        log(
            INFO,
            f"server-side full dataset evaluation Round {server_round} - Evaluation loss: {loss}, accuracy: {accuracy}",
        )
        # writing logs to file
        file_save_path = get_save_path(args, "loss", ".png", with_time=False)
        draw_curve(metrics["loss"], "loss", file_save_path)
        file_save_path = get_save_path(args, "acc", ".png", with_time=False)
        draw_curve(metrics["acc"], "acc", file_save_path)

        log(
            INFO,
            f"-------------------------------evaluation-----------------------------",
        )
        return loss, {"accuracy": accuracy}

    return evaluate


def parse_args(parser):
    global EPOCHS, SEED, NUM_CLIENTS, BATCH_SIZE, MOMENTUM, LEARNIGN_RATE, WEIGHT_DECAY
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        help="compression mode in 'baseline, topk, randomk'",
    )
    parser.add_argument("--rounds", type=int, default=EPOCHS, help="number of rounds")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    parser.add_argument(
        "--num_clients", type=int, default=NUM_CLIENTS, help="number of clients"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="batch size for training"
    )
    parser.add_argument(
        "--momentum", type=float, default=MOMENTUM, help="momentum for SGD optimizer"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LEARNIGN_RATE, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=WEIGHT_DECAY, help="weight decay"
    )

    args = parser.parse_args()
    NUM_CLIENTS = args.num_clients
    BATCH_SIZE = args.batch_size
    MOMENTUM = args.momentum
    LEARNIGN_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.rounds
    SEED = args.seed

    return args


# Global variables to store metrics and final results
metrics = {"round": [], "acc": [], "loss": []}
final_results = {"acc": [], "data": [], "time": []}
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="baseline args")
    args = parse_args(parser)
    prefix, timestamp = get_task_prefix(args)
    log_name = "_".join(prefix + [timestamp + ".logs"])
    fl.common.logger.configure(
        identifier="BaselineExperiment", filename=LOG_DIR / log_name
    )
    log(INFO, f"Experiment logs will be saved to {LOG_DIR / log_name}")
    log(INFO, f"Experiment settings: {args}")
    if args.mode == "baseline":
        client_compr_fn = ndarrays_to_parameters

    elif args.mode == "topk":
        from compr.topk import topk_ndarrays_to_parameters

        client_compr_fn = topk_ndarrays_to_parameters

    elif args.mode == "randomk":
        from compr.randomk import randomk_ndarrays_to_parameters

        client_compr_fn = randomk_ndarrays_to_parameters

    # default compr & uncompr fn
    client_uncompr_fn = parameters_to_ndarrays
    server_compr_fn, server_uncompr_fn = (
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    # Load data
    trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
    # Collection partitioned data distribution
    data_patitions = collect_data_partition(args, trainloaders, num_classes=10)
    draw_data_partition(args, data_patitions)

    # Load model
    net = get_model()

    # Define client_fn
    def client_fn(cid) -> FlowerClient:
        net = get_model().to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(
            cid,
            net,
            trainloader,
            valloader,
            uncompr_fn=client_uncompr_fn,
            compr_fn=client_compr_fn,
            log_name=log_name,
        )

    # Define strategy
    strategy = FedBaseline(
        initial_parameters=ndarrays_to_parameters(get_parameters(net)),
        evaluate_fn=get_evaluate_fn(net=net, testloader=testloader),
        uncompr_fn=server_uncompr_fn,
        compr_fn=server_compr_fn,
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_cpus": 4, "num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        strategy=strategy,
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        client_resources=client_resources,
        ray_init_args={"num_cpus": 8, "num_gpus": 2},
    )
    # Save results
    save_results(args, metrics, final_results)
