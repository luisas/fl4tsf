"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO

import torch
import wandb
from flower.task import Net, create_run_dir, set_weights

from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from functools import partial, reduce
import numpy as np
from flower.model_config import get_model_config

from flwr.common import (
    NDArrays,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from typing import Callable, Optional, Union
from flwr.server.strategy.aggregate import aggregate

PROJECT_NAME = "FLOWER-advanced-pytorch"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.aggregate_fun_name = get_model_config(file_path="model.config")["aggregation"]
        self.alpha = float(get_model_config(file_path="model.config")["alpha"])
        self.use_wandb = use_wandb
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, round, accuracy, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        model = Net()
        set_weights(model, ndarrays)
        # Save the PyTorch model
        file_name = f"model.pth"
        torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        print(f"Results: {results}")

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
    

    #############################################################
    # Update the aggregate function for the latent ODE model
    #############################################################

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
        ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics["nodesolve"])
            for _, fit_res in results
        ]

        if(self.aggregate_fun_name == "FedAvg"):
            # Aggregate using average
            parameters_aggregated = aggregate_avg(weights_results)
        elif(self.aggregate_fun_name == "FedODE"):
            # Aggregate using ODE
            parameters_aggregated = aggregate_ode(weights_results, self.alpha)
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregate_fun_name}")    
           
        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

def aggregate_ode(results: list[tuple[NDArrays, int]], alpha =0.5) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples, _) in results)

    # Results are a list of tuples (weights, num_examples)
    # the length of results is the number of clients
    # Each element of results is a tuple (weights, num_examples)
    # Weights are a list of NDArrays


    # # alpha controls balance between number of examples and ODE steps
    
    # Extract num_examples and num_steps from results
    num_examples_list = [num_examples for _, num_examples, _ in results]
    num_steps_list = [num_steps for _, _, num_steps in results]

    # Total sums for normalization
    total_examples = sum(num_examples_list)
    total_steps = sum(num_steps_list)

    # Compute lambda_k for each client
    lambdas = [
        alpha * (n / total_examples) + (1 - alpha) * (m / total_steps)
        for n, m in zip(num_examples_list, num_steps_list)
    ]

    # Weighted model parameters using lambda_k
    weighted_weights = [
        [layer * lam for layer in weights]
        for (weights, lam) in zip([r[0] for r in results], lambdas)
    ]

    # Aggregate across clients
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


def aggregate_avg(results: list[tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples, _) in results)


    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples, _ in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime