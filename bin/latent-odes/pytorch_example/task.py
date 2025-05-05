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



def Net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obsrv_std = 0.01
    poisson = False
    units = 100
    latents = 10
    gen_layers = 1
    rec_dims = 20
    rec_layers = 1
    z0_encoder = "rnn"
    gru_units = 3
    train_classif_w_reconstr = False
    classif = False
    linear_classif = False
    classif_per_tp = False
    n_labels = 1
    input_dim = 1
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


def train(net, trainloader, epochs, lr, device, loss_per_epoch=False):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    
    optimizer = optim.Adamax(net.parameters(), lr=lr)

    #num_batches = trainloader.__len__()

    #print("Number of batches: ", num_batches)

    # wait_until_kl_inc = 10
    # if epochs // num_batches < wait_until_kl_inc:
    #     kl_coef = 0.
    # else:
    #     kl_coef = (1-0.99** (epochs // num_batches - wait_until_kl_inc))
    kl_coef = 0.1

    running_loss = 0.0
    if loss_per_epoch:
        epoch_loss = []

    n_batches = len(trainloader)
    print("Number of batches: ", n_batches)
    trainloader = utils.inf_generator(trainloader)
    
    for _ in range(epochs): # TODO 
        # for batch in trainloader:
        # print the first element of the batch
        optimizer.zero_grad()
        batch_dict = utils.get_next_batch(trainloader)
        train_res = net.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()
        optimizer.step()
        loss = train_res["loss"].item()
        running_loss += loss
        if loss_per_epoch:
            epoch_loss.append(loss.item())
    avg_trainloss = 10 #todo
    return avg_trainloss

def test(net, testloader, device):    
    """Validate the model on the test set."""
    net.to(device)
    # # print how big the testloader is 
    # correct, loss = 0, 0.0
    # with torch.no_grad():
    #     for batch in testloader:
    #         # Fix kl_coef to 0 for testing
    #         test_res = net.compute_all_losses(batch, n_traj_samples = 1, kl_coef = 0)
    #         loss += test_res["loss"].item()
    # accuracy = 0 # TODO implement accuracy
    # loss = loss / len(testloader)
    loss = 0.0
    accuracy = 0.0
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def create_periodic_dataset():
    """Create a periodic dataset."""
    args = SimpleNamespace()
    args.dataset = "periodic"
    args.extrap = False
    args.timepoints = 5
    args.max_t = 5.
    args.n = 100
    args.noise_weight = 0.1
    args.batch_size = 32
    args.quantization = 0.1
    args.classif = False
    args.sample_tp = 0.5
    args.cut_tp = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, timestamps, basic_collate_fn = parse_datasets(args, device)

    return  dataset, timestamps, basic_collate_fn

dataset = None # Cache dataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition of periodic dataset for federated learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the full periodic dataset
    batch_size = 32 # TODO 
    n = 100 # TODO 

    # 0. Create the dataset (if not already created)  
    global dataset
    global time_steps_extrap
    global basic_collate_fn
    if dataset is None:
        dataset, time_steps_extrap, basic_collate_fn = create_periodic_dataset()

    # 1. Extract the partition
    partition_len = len(dataset) // num_partitions
    start = partition_id * partition_len
    end = (partition_id + 1) * partition_len if partition_id < num_partitions - 1 else len(dataset)
    partition_dataset = dataset[start:end, :, :]
    if len(partition_dataset) == 0:
        raise ValueError(f"Partition {partition_id} has 0 samples. Adjust num_partitions or dataset size.")

    # 2. Split into train and test
    train_y, test_y = utils.split_train_test(partition_dataset, train_fraq = 0.8)

    train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
    test_dataloader = DataLoader(test_y, batch_size = n, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))

    return train_dataloader, test_dataloader



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