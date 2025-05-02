
import torch
from torch.utils.data import DataLoader
from task import Net, load_data, train, test
import pandas as pd
import os

# load model from pth 
def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    return model

model = Net()
model = load_model(model, "../outputs/federated_outputs/model.pth")