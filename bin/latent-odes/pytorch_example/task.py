"""pytorch-example: A Flower / PyTorch app."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


from torchdiffeq import odeint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from flwr.common.typing import UserConfig


from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver



def Net(device):
    obsrv_std = 0.01
    poisson = False
    units = 100
    latents = 10
    gen_layers = 1
    rec_dims = 20
    rec_layers = 1
    z0_encoder = "classic_rnn"
    train_classif_w_reconstr = False
    classif = False
    linear_classif = False
    classif_per_tp = False
    n_labels = 1
    input_dim = 28 * 28
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model = create_LatentODE_model(latents,
                                    poisson,
                                    units,
                                    gen_layers,
                                    rec_dims,
                                    rec_layers,
                                    z0_encoder,
                                    classif,
                                    linear_classif,
                                    train_classif_w_reconstr,
                                    input_dim,
                                    z0_prior,
                                    obsrv_std,
                                    device, 
                                    classif_per_tp = classif_per_tp,
                                    n_labels = n_labels)
    return model


def train(net, trainloader, epochs, lr, device, loss_per_epoch=False):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    
    optimizer = optim.Adamax(net.parameters(), lr=lr)
    num_batches = trainloader.__len__()

    wait_until_kl_inc = 10
    if epochs // num_batches < wait_until_kl_inc:
        kl_coef = 0.
    else:
        kl_coef = (1-0.99** (epochs // num_batches - wait_until_kl_inc))

    if loss_per_epoch:
        epoch_loss = []
    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            #utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = lr / 10)
            train_res = net.compute_all_losses(batch.to(device), n_traj_samples = 3, kl_coef = kl_coef)
            train_res["loss"].backward()
            optimizer.step()
            loss = train_res["loss"].item()
            running_loss += loss
        if loss_per_epoch:
            epoch_loss.append(loss.item())
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

# TODO implement 
# def test(net, testloader, device):    
#     """Validate the model on the test set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["image"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


fds = None  # Cache FederatedDataset

# TODO: use time series data instead of MNIST 
def load_data(partition_id: int, num_partitions: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=1.0,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    # Print here how long the dataset is 
    total = sum(len(fds.load_partition(i)) for i in range(num_partitions))
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"]
    test_partition = partition_train_test["test"]
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = "federated_outputs"
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir