
import torch
from torch.utils.data import DataLoader
from lib.parse_datasets import parse_datasets
from types import SimpleNamespace
from lib import utils
import os


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

