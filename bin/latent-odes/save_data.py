###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

from types import SimpleNamespace
import os
import numpy as np
import torch
from lib.parse_datasets import parse_datasets


if __name__ == '__main__':

	print("Generating periodic time series...")

	args = SimpleNamespace()
	args.dataset = "periodic"
	args.extrap = False
	args.timepoints = 5
	args.max_t = 5.
	args.n = 2
	args.noise_weight = 0.1
	args.batch_size = 32
	args.quantization = 0.1
	args.classif = False
	args.sample_tp = 0.5
	args.cut_tp = None
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	data_obj = parse_datasets(args, device)

	# Extract data loader 
	trainloader = data_obj["train_dataloader"]
	testloader = data_obj["test_dataloader"]
	# extract one element from the data loader and print it
	for batch in trainloader:
		print("Batch: ", batch)
		break

	print("-------------------")

	# extract one element from the data loader and print it
	for batch in testloader:
		print("Batch: ", batch)
		break
	print("Data object: ", data_obj)