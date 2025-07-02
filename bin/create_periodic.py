#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
from lib.parse_datasets import parse_datasets
from types import SimpleNamespace
from lib import utils
import os
from sklearn import model_selection

# Creates a periodic dataset and stores it in the specified directory
# Toy example dataset for Rubanova et al. Latent ODEs
def store_periodic_dataset(args):
	"""Create a periodic dataset."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset, time_steps_extrap, _ = parse_datasets(args, device)

	# Add one dimention to the time_stes to make it compatible with the dataset
	time_steps_extrap = time_steps_extrap.unsqueeze(0)

	# 2. Split into train and test
	c0, c1 = utils.split_train_test(dataset, train_fraq = 0.5)

	train_0, test_0 = utils.split_train_test(c0, train_fraq = 0.8)
	train_1, test_1 = utils.split_train_test(c1, train_fraq = 0.8)

	# store the train and test datasets
	output_file_prefix = "../data/periodic_clients/"
	# if it doesn't exist, create the directory
	if not os.path.exists(f"{output_file_prefix}"):
		os.makedirs(f"{output_file_prefix}", exist_ok=True)

	# save the dataset to a file of first client
	with open(f"{output_file_prefix}/client_0_train.pt", "wb") as f:
		torch.save(train_0, f)
	with open(f"{output_file_prefix}/client_0_test.pt", "wb") as f:
		torch.save(test_0, f)
	with open(f"{output_file_prefix}/client_0_time_steps_train.pt", "wb") as f:
		torch.save(time_steps_extrap, f)
	with open(f"{output_file_prefix}/client_0_time_steps_test.pt", "wb") as f:
		torch.save(time_steps_extrap, f)

	# save the dataset to a file of second client
	with open(f"{output_file_prefix}/client_1_train.pt", "wb") as f:
		torch.save(train_1, f)
	with open(f"{output_file_prefix}/client_1_test.pt", "wb") as f:
		torch.save(test_1, f)
	with open(f"{output_file_prefix}/client_1_time_steps_train.pt", "wb") as f:
		torch.save(time_steps_extrap, f)
	with open(f"{output_file_prefix}/client_1_time_steps_test.pt", "wb") as f:
		torch.save(time_steps_extrap, f)

    
	# print message with colors
	print("\033[92mPeriodic dataset created and saved to disk.\033[0m")
	print(f"{output_file_prefix}_train.pt")
	print(f"{output_file_prefix}_test.pt")
	print(f"{output_file_prefix}_time_steps.pt")


if __name__ == "__main__":
	args = SimpleNamespace()
	args.dataset = "periodic"
	args.extrap = False
	args.timepoints = 100
	args.max_t = 5.
	args.noise_weight = 0.01
	args.n = 2000
	store_periodic_dataset(args)