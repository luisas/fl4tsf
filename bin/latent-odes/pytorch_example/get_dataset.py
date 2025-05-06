
import torch
from torch.utils.data import DataLoader
from lib.parse_datasets import parse_datasets
from types import SimpleNamespace
from lib import utils
import os

def get_dataset(dataset_name, type, data_folder = "../../data/"):
    """
    Load the dataset from the specified folder.
    """
    # Check if the dataset exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Dataset folder {data_folder} does not exist.")
    
    # Load the dataset
    dataset = torch.load(os.path.join(data_folder, f"{dataset_name}/{dataset_name}_{type}.pt"), weights_only=True)
    timestamps = torch.load(os.path.join(data_folder, f"{dataset_name}/{dataset_name}_time_steps.pt"), weights_only=True)
    
    return dataset, timestamps


def basic_collate_fn(batch, time_steps, dataset_name, sample_tp, cut_tp, extrap, data_type = "train"):

    args = SimpleNamespace()
    args.dataset = dataset_name
    args.sample_tp = sample_tp
    args.cut_tp = cut_tp
    args.extrap = extrap

    batch = torch.stack(batch)
    data_dict = {
        "data": batch, 
        "time_steps": time_steps}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict

