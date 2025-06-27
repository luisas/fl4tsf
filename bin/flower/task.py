"""pytorch-example: A Flower / PyTorch app."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace
from flwr.common.typing import UserConfig

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from lib.parse_datasets import parse_datasets
from lib.collate_functions import basic_collate_fn
from flower.model_config import get_model_config
from lib.physionet import variable_time_collate_fn

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


def train(net, trainloader, valloader, epochs, lr, device, loss_per_epoch=True ):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    net.train()  # Set the model to training mode
    n_batches = len(trainloader)
    optimizer = optim.AdamW(net.parameters(), lr=lr)

    model_config = get_model_config(file_path="model.config")

    running_loss = 0.0
    if loss_per_epoch:
        epoch_loss = []
        epoch_mse = []
        val_loss = []
        val_mse = []
        lrs = []
    

    # Track the number of steps that the solver has taken
    nodesolves = []
    nodesolves_epoch = []
    trainloader = utils.inf_generator(trainloader)

    # So that we can use the same kl_coef for training and testing
    n_total_iters = epochs * n_batches
    global kl_coef 
    kl_coef = 0.993
    for itr in range(1, n_total_iters + 1):

        # Set the epoch
        epoch = itr // n_batches
        
        # Reset gradients
        optimizer.zero_grad()

        # Learning rate decay
        decay_rate = float(model_config["lrdecay"])

        # KL annealing
        wait_until_kl_inc = 10
        if epoch < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (epoch - wait_until_kl_inc))

        # Get the next batch
        batch_dict = utils.get_next_batch(trainloader)

        # Compute the loss
        batch_dict = utils.move_to_device(batch_dict, device)
        train_res = net.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()

        # Collect gradients
        grad_norms = []
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad_norms.append((name, param.grad.data.norm(2).item()))

        # Clip gradients
        if model_config["gradientclipping"] == "True":
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        # Update weights
        optimizer.step()

        # Collect all metrics
        loss = train_res["loss"].item()
        mse = train_res["mse"].item()
        pois_likelihood = train_res["pois_likelihood"].item()
        ce_loss = train_res["ce_loss"].item()
        kl_first_p = train_res["kl_first_p"].item()
        std_first_p = train_res["std_first_p"].item()
        nodesolve = train_res["nodesolve"]
        nodesolves.append(nodesolve)
        running_loss += loss

        if itr % n_batches == 0:
            if loss_per_epoch:
                epoch_loss.append(loss)
                epoch_mse.append(mse)
            print(f"Epoch {itr // n_batches} / {epochs}, loss: {loss:.4f}, mse: {train_res['mse'].item():.4f}, kl_coef: {kl_coef:.4f}, pois_likelihood: {pois_likelihood:.4f}, ce_loss: {ce_loss:.4f}, kl_first_p: {kl_first_p:.4f}, std_first_p: {std_first_p:.4f}")
            # store the weights of the model
            if decay_rate < 1.0 and itr > 1:    
                utils.update_learning_rate(optimizer, decay_rate = decay_rate, lowest = lr/10)
            lrs.append(optimizer.param_groups[0]['lr'])
            # Validation evaluation
            if valloader is not None:
                val_l, val_m = test(net, valloader, device)
                val_loss.append(val_l)
                val_mse.append(val_m)

    avg_trainloss = running_loss/n_batches

    # print weights
    file_store = None
    if(model_config["storeweights"] == "True"):
        w = get_weights(net)
        # store them 
        random_id = str(int(torch.randint(0, 1000000, (1,)).item()))
        file_store = f"weights_{random_id}.pt"
        torch.save(w, file_store)
    
    
    dict_metrics = {
        "train_loss": epoch_loss,
        "train_mse": epoch_mse,
        "val_loss": val_loss if loss_per_epoch else None,
        "val_mse": val_mse if loss_per_epoch else None,
        "nodesolves": nodesolves,
        "weights": file_store,
        "grad_norms": grad_norms,
        "lr": lrs

    }

    return avg_trainloss, sum(nodesolves), dict_metrics


def test(net, dataloader, device, kl_coef = 1.0):    
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    n_batches = len(dataloader)
    dataloader = utils.inf_generator(dataloader)

    total_loss = 0.0
    total_mse = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            batch_dict = utils.get_next_batch(dataloader)
            batch_dict = utils.move_to_device(batch_dict, device)
            res = net.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
            total_loss += res["loss"].item()
            total_mse += res["mse"].item()

    avg_loss = total_loss / n_batches
    avg_mse = total_mse / n_batches

    return avg_loss, avg_mse


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

    model_config = get_model_config(file_path="model.config")
    dataset_name = model_config["dataset_name"]
    sample_tp = float(model_config["sample_tp"])
    cut_tp = None
    extrap = False
    data_folder = model_config["data_folder"]

    # load partitioned dataset
    partition_name = f"client_{partition_id}"

    train_dataset = torch.load(os.path.join(data_folder, f"{partition_name}_train.pt"), weights_only=True)
    test_dataset = torch.load(os.path.join(data_folder, f"{partition_name}_test.pt"), weights_only=True)

    if "physionet" in dataset_name:
        from types import SimpleNamespace
        args = SimpleNamespace()
        args.sample_tp = sample_tp
        args.cut_tp = cut_tp
        args.extrap = extrap
        data_min = torch.load(os.path.join(data_folder, f"{partition_name}_data_min.pt"), weights_only=True)
        data_max = torch.load(os.path.join(data_folder, f"{partition_name}_data_max.pt"), weights_only=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(batch, args, device, data_type="train",
                data_min=data_min, data_max=data_max))
        validation_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
                data_min = data_min, data_max = data_max))

    else:    

        timesteps_train = torch.load(os.path.join(data_folder, f"{partition_name}_time_steps_train.pt"), weights_only=True)
        timesteps_test = torch.load(os.path.join(data_folder, f"{partition_name}_time_steps_test.pt"), weights_only=True)
        # take the first element of timestep tensors
        timesteps = timesteps_train[0]

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: basic_collate_fn(batch, timesteps, dataset_name, sample_tp, cut_tp, extrap, data_type = "train"))
        validation_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: basic_collate_fn(batch, timesteps, dataset_name, sample_tp, cut_tp, extrap, data_type = "test"))

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