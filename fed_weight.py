import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from ser.sparse import ndarrays_to_sparse_parameters, sparse_parameters_to_ndarrays

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

NUM_CLIENTS = 2
# set client device using gpu or cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


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


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


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
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)


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
        uncompr_fn=parameters_to_ndarrays,
        compr_fn=ndarrays_to_parameters,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.uncompr_fn = uncompr_fn
        self.compr_fn = compr_fn

    # def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
    #     print(f"[Client {self.cid}] get_parameters")

    #     # Get parameters as a list of NumPy ndarray's
    #     ndarrays: List[np.ndarray] = get_parameters(self.net)

    #     # Serialize ndarray's into a Parameters object using our custom function
    #     parameters = self.compr_fn(ndarrays)

    #     bytes = sum([p.size * p.itemsize for p in ndarrays])
    #     print(f"[Client {self.cid}] parameters byte size: {bytes}")
    #     # Build and return response
    #     status = Status(code=Code.OK, message="Success")
    #     return GetParametersRes(
    #         status=status,
    #         parameters=parameters,
    #     )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = self.uncompr_fn(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(self.net, self.trainloader, epochs=1)
        ndarrays_updated = get_parameters(self.net)
        np.savez(f"client_{self.cid}_parameters.npz", *ndarrays_updated)
        bytes = sum([p.nbytes for p in ndarrays_updated])
        print(
            f"-------------------------------data saved for client {self.cid}-----------------------------"
        )
        print(f"[Client {self.cid}] ndarrays byte size: {bytes}")
        # Serialize ndarray's into a Parameters object using our custom function
        parameters_updated = self.compr_fn(ndarrays_updated)
        # import pickle

        # # Save parameters_updated to a file
        # with open("parameters_updated.pkl", "wb") as f:
        #     pickle.dump(parameters_updated, f, protocol=2)
        total_bytes = 0
        for p in parameters_updated.tensors:
            total_bytes += len(p)
        print(f"[Client {self.cid}] compressed parameters byte size: {total_bytes}")
        print(
            f"-------------------------------data saved for client {self.cid}-----------------------------"
        )
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's using our custom function
        parameters_original = ins.parameters
        ndarrays_original = self.uncompr_fn(parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


# define the server side
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
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
        parameters_ndarrays = self.uncompr_fn(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
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
        print(
            f"-------------------------------weights_results-----------------------------"
        )
        for parameters, num_examples in weights_results:
            print(f"num_examples: {num_examples}")
            for p in parameters:
                print(f"p size: {p.shape}")

        # We serialize the aggregated result using our cutom method
        parameters_aggregated = self.compr_fn(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        self.parameters_aggregated = parameters_aggregated
        return parameters_aggregated, metrics_aggregated


def get_evaluate_fn(net, testloader):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(DEVICE)
        # save net
        DATA_DIR = Path("./data")
        torch.save(net, DATA_DIR / f"server_round_{server_round}_net.pth")
        total_loss, total_correct, total_samples = 0, 0, 0

        with torch.no_grad():  # Do not calculate gradients
            for data, target in testloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = net(data)  # Forward pass
                loss = F.cross_entropy(
                    output, target, reduction="sum"
                )  # Calculate loss
                pred = output.argmax(dim=1)  # Get predictions
                correct = (
                    pred.eq(target.view_as(pred)).sum().item()
                )  # Calculate correct predictions

                total_loss += loss.item()
                total_correct += correct
                total_samples += data.shape[0]

        loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        log(
            INFO,
            f"server-side full dataset evaluation Round {server_round} - Evaluation loss: {loss}, accuracy: {accuracy}",
        )
        print(f"-------------------------------evaluation-----------------------------")
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--mode",
        type=str,
        default="weight",
        help="compression mode in 'weight, topk, randomk'",
    )

    args = parser.parse_args()

    if args.mode == "weight":
        # default mode compression and uncompression function
        client_compr_fn, client_uncompr_fn = (
            ndarrays_to_parameters,
            parameters_to_ndarrays,
        )
        server_compr_fn, server_uncompr_fn = (
            ndarrays_to_parameters,
            parameters_to_ndarrays,
        )

    elif args.mode == "topk":
        from compr.topk import topk_ndarrays_to_parameters, topk_parameters_to_ndarrays

        client_compr_fn, client_uncompr_fn = (
            topk_ndarrays_to_parameters,
            parameters_to_ndarrays,
        )
        server_compr_fn, server_uncompr_fn = (
            ndarrays_to_parameters,
            topk_parameters_to_ndarrays,
        )
    elif args.mode == "randomk":
        from compr.randomk import (
            randomk_ndarrays_to_parameters,
            randomk_parameters_to_ndarrays,
        )

        client_compr_fn, client_uncompr_fn = (
            randomk_ndarrays_to_parameters,
            parameters_to_ndarrays,
        )
        server_compr_fn, server_uncompr_fn = (
            ndarrays_to_parameters,
            randomk_parameters_to_ndarrays,
        )

    net = Net()

    def client_fn(cid) -> FlowerClient:
        net = Net().to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(
            cid,
            net,
            trainloader,
            valloader,
            uncompr_fn=client_uncompr_fn,
            compr_fn=client_compr_fn,
        )

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

    fl.simulation.start_simulation(
        strategy=strategy,
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        client_resources=client_resources,
        ray_init_args={"num_cpus": 8, "num_gpus": 2},
    )
