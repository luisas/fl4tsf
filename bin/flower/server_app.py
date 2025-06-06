"""pytorch-example: A Flower / PyTorch app."""

import torch
from flower.strategy import CustomFedAvg
from flower.task import (
    Net,
    get_weights,
    set_weights,
    test,
)
from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from types import SimpleNamespace
from flower.get_dataset import get_dataset, basic_collate_fn
import lib.utils as utils
from flower.model_config import get_model_config

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""


    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device=device, kl_coef = 0.993)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    model_config = get_model_config(file_path="model.config")
    lr = float(model_config["lr"])
    decay_onset = int(model_config["decay_onset"])
    # Enable a simple form of learning rate decay
    if server_round > decay_onset:
        # Reduce learning rate by 10% for each round after the first
        lr /= 10
    return {"lr": lr}


# Define metric aggregation function
def weighted_average(metrics):

    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    #losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context, nrounds: int = 4):
    # Read from config
    model_config = get_model_config(file_path="model.config")

    num_rounds = int(model_config["serverrounds"])
    fraction_fit = float(model_config["fractionfit"])
    fraction_eval = float(model_config["fractionevaluate"])
    server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_config = {
        "num-server-rounds": num_rounds,
        "fraction-fit": fraction_fit,
        "fraction-evaluate": fraction_eval
    }

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    dataset_name = model_config["dataset_name"]
    sample_tp = float(model_config["sample_tp"])
    cut_tp = None
    extrap = False
    batch_size = int(model_config["batch_size"])
    extrap = bool(model_config["extrap"])
    data_folder = model_config["data_folder"]

    # Identify partitions 
    partitions = {
        "_".join(f.split("_")[:2])
        for f in os.listdir(data_folder)
        if f.startswith("client") and f.endswith("test.pt")
    }

    print(f"Found partitions: {sorted(partitions)}")

    test_dataset = torch.cat([
        torch.load(f"{p}_test.pt", weights_only=True) for p in partitions
    ], dim=0)

    test_timestamps = torch.cat([
        torch.load(f"{p}_time_steps_test.pt", weights_only=True) for p in partitions
    ], dim=0)


    testloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, test_timestamps, dataset_name, sample_tp, cut_tp, extrap, data_type = "test"))

    strategy = CustomFedAvg(
        run_config=run_config,
        use_wandb=model_config["use_wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config, )
