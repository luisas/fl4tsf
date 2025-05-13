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
from types import SimpleNamespace

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from lib.parse_datasets import parse_datasets
from flower.get_dataset import get_dataset, basic_collate_fn
from flower.model_config import get_model_config

def Net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = get_model_config(file_path="model.config")

    obsrv_std = float(model_config["obsrv_std"])
    poisson = model_config["poisson"] == "True"
    rec_layers = int(model_config["rec_layers"])
    gen_layers = int(model_config["gen_layers"])
    units = int(model_config["units"])
    gru_units = int(model_config["gru_units"])
    latents = int(model_config["latents"])
    rec_dims = int(model_config["rec_dims"])
    z0_encoder = model_config["z0_encoder"]
    train_classif_w_reconstr = model_config["train_classif_w_reconstr"] == "True"
    classif = model_config["classif"] == "True"
    linear_classif = model_config["linear_classif"] == "True"
    classif_per_tp = model_config["classif_per_tp"] == "True"
    n_labels = int(model_config["n_labels"])
    input_dim = int(model_config["input_dim"])
    
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model = create_LatentODE_model(latents,
                                    poisson,
                                    units,
                                    gen_layers,
                                    rec_dims,
                                    rec_layers,
                                    gru_units,
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

kl_coef = 0.0

def train(net, trainloader, epochs, lr, device, loss_per_epoch=False):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    n_batches = len(trainloader)
    optimizer = optim.Adamax(net.parameters(), lr=lr)
    
    running_loss = 0.0
    if loss_per_epoch:
        epoch_loss = []
        epoch_mse = []
    
    # Track the number of steps that the solver has taken
    nodesolves = []
    trainloader = utils.inf_generator(trainloader)
    # So that we can use the same kl_coef for training and testing
    global kl_coef 
    kl_coef = 0.9
    for itr in range(1, (n_batches * epochs) + 1):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = lr / 10)
        # wait_until_kl_inc = 10
        # if itr // n_batches < wait_until_kl_inc:
        #     kl_coef = 0.
        # else:
        #     kl_coef = (1-0.99** (epochs // n_batches - wait_until_kl_inc))
        batch_dict = utils.get_next_batch(trainloader)
        train_res = net.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
 
        train_res["loss"].backward()
        optimizer.step()
        loss = train_res["loss"].item()
        mse = train_res["mse"].item()
        pois_likelihood = train_res["pois_likelihood"].item()
        ce_loss = train_res["ce_loss"].item()
        kl_first_p = train_res["kl_first_p"].item()
        std_first_p = train_res["std_first_p"].item()
        nodesolve = train_res["nodesolve"]
        running_loss += loss
        nodesolves.append(nodesolve)

        if itr % n_batches == 0:
            if loss_per_epoch:
                epoch_loss.append(loss)
                epoch_mse.append(mse)
            print(f"Epoch {itr // n_batches} / {epochs}, loss: {loss:.4f}, mse: {train_res['mse'].item():.4f}, kl_coef: {kl_coef:.4f}, pois_likelihood: {pois_likelihood:.4f}, ce_loss: {ce_loss:.4f}, kl_first_p: {kl_first_p:.4f}, std_first_p: {std_first_p:.4f}")
    avg_trainloss = running_loss/n_batches
    print(nodesolves)
    if loss_per_epoch:
        return avg_trainloss, epoch_loss, epoch_mse, sum(nodesolves)
    return avg_trainloss, None, None, sum(nodesolves)

def test(net, testloader, device):    
    """Validate the model on the test set."""
    net.to(device)
    n_batches = len(testloader)
    testloader = utils.inf_generator(testloader)
    with torch.no_grad():
        for itr in range(1,n_batches+1):
            batch_dict = utils.get_next_batch(testloader)
            test_res = net.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
            loss = test_res["loss"].item()
            mse = test_res["mse"].item()
    loss = loss / n_batches
    mse = mse / n_batches
    return loss, mse


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


dataset = None # Cache dataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition of periodic dataset for federated learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the full periodic dataset
    # 0. Create the dataset (if not already created)

    global dataset
    global time_steps_extrap
    global basic_collate_fn
    global sample_tp
    global cut_tp
    global extrap
    global data_folder
    global dataset_name

    if dataset is None:
        model_config = get_model_config(file_path="model.config")
        dataset_name = model_config["dataset_name"]
        sample_tp = float(model_config["sample_tp"])
        cut_tp = None
        extrap = False
        data_folder = model_config["data_folder"]
        dataset, time_steps_extrap = get_dataset(dataset_name = dataset_name, type="train", data_folder=data_folder)


    # 1. Extract the partition
    partition_len = len(dataset) // num_partitions
    start = partition_id * partition_len
    end = (partition_id + 1) * partition_len if partition_id < num_partitions - 1 else len(dataset)
    partition_dataset = dataset[start:end, :, :]
    if len(partition_dataset) == 0:
        raise ValueError(f"Partition {partition_id} has 0 samples. Adjust num_partitions or dataset size.")

    train_dataset, validation_dataset = utils.split_train_test(partition_dataset, train_fraq = 0.8)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, data_type = "train"))
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, data_type = "test"))

    return train_loader, validation_loader


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = "federated_outputs"
    # Save path is based on the current directory
    save_path = Path.cwd() / f"{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir