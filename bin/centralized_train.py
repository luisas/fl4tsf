#!/usr/bin/env python

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from flower.task import Net, train
from lib.collate_functions import basic_collate_fn
from lib.physionet import variable_time_collate_fn, get_data_min_max



def get_parameters():
    import argparse
    parser = argparse.ArgumentParser(description='Train a centralized model.')
    parser.add_argument('--dataset', type=str, default="periodic", help='Dataset name')
    parser.add_argument('--sample_tp', type=float, default=0.5, help='Sample time period')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name')
    parser.add_argument('--output_dir', type=str, default=".", help='Output directory')
    args = parser.parse_args()
    return args

args = get_parameters()
if args.dataset_name is not None:
    dataset_name = args.dataset_name
else:
    dataset_name = args.dataset
sample_tp = args.sample_tp
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


# Fix 
cut_tp = None
extrap = False
data_folder = "."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and load data
model = Net()

train_dataset     = torch.load(f"{data_folder}/{dataset_name}_train.pt", weights_only=False)
test_dataset      = torch.load(f"{data_folder}/{dataset_name}_test.pt", weights_only=False)

if "physionet" in args.dataset or "ecg" in args.dataset:
    print("Using PhysioNet dataset with variable time steps.")
    from types import SimpleNamespace
    args_physionet = SimpleNamespace()
    args_physionet.sample_tp = sample_tp
    args_physionet.cut_tp = None
    args_physionet.extrap = None
    if os.path.exists(f"{data_folder}/{dataset_name}_data_min.pt"):
        data_min = torch.load(f"{data_folder}/{dataset_name}_data_min.pt", weights_only=False)
        data_max = torch.load(f"{data_folder}/{dataset_name}_data_max.pt", weights_only=False)
    else:
        total_dataset = train_dataset + test_dataset
        data_min, data_max = get_data_min_max(total_dataset)
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False, 
        collate_fn= lambda batch: variable_time_collate_fn(batch, args_physionet, device, data_type = "train",
            data_min = data_min, data_max = data_max))
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, args_physionet, device, data_type = "test",
            data_min = data_min, data_max = data_max))
else:
    time_steps_extrap = torch.load(f"{data_folder}/{dataset_name}_time_steps_train.pt", weights_only=False)
    time_steps_extrap = time_steps_extrap[0]
    # train dataset 
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, device, data_type = "train"))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, dataset_name, sample_tp, cut_tp, extrap, device, data_type = "test"))

# train
loss_training = train(model, train_loader, test_loader, epochs, lr=lr, device=device, loss_per_epoch=True)

# Store the model 
torch.save(model.state_dict(), "model.pth")

# #######################################
# Store files
# #######################################

avg_loss, nodesolves, metric_dict = loss_training
train_loss = metric_dict["train_loss"]
train_mse = metric_dict["train_mse"]
val_loss = metric_dict["val_loss"]
val_mse = metric_dict["val_mse"]

print("Nodesolves:", nodesolves)
print("Train loss:", train_loss)

df = pd.DataFrame({"train_loss": train_loss, "train_mse": train_mse, "val_loss": val_loss, "val_mse": val_mse, "nodesolves": nodesolves})

# if output_dir does not exist, create it
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# add epochs and header
df.index.name = "epoch"
# add output directory
df.to_csv(os.path.join(args.output_dir, "loss_per_epoch.csv"), index=True)