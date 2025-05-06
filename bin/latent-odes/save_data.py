
import torch
from torch.utils.data import DataLoader
from lib.parse_datasets import parse_datasets
from types import SimpleNamespace
from lib import utils
import os


# Creates a periodic dataset and stores it in the specified directory
# Toy example dataset for Rubanova et al. Latent ODEs
def store_periodic_dataset():
	"""Create a periodic dataset."""

	args = SimpleNamespace()
	args.dataset = "periodic"
	args.extrap = False
	args.timepoints = 100
	args.max_t = 5.
	args.noise_weight = 0.01
	args.n = 1000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset, time_steps_extrap, _ = parse_datasets(args, device)

	# 2. Split into train and test
	train, test = utils.split_train_test(dataset, train_fraq = 0.8)


	# store the train and test datasets
	output_file_prefix = "../../data/periodic/periodic"
	# if it doesn't exist, create the directory
	if not os.path.exists("../data/periodic"):
		os.makedirs("../../data/periodic", exist_ok=True)

	# save the dataset to a file
	with open(f"{output_file_prefix}_train.pt", "wb") as f:
		torch.save(train, f)
	with open(f"{output_file_prefix}_test.pt", "wb") as f:
		torch.save(test, f)
	with open(f"{output_file_prefix}_time_steps.pt", "wb") as f:
		torch.save(time_steps_extrap, f)

    
	# print message with colors
	print("\033[92mPeriodic dataset created and saved to disk.\033[0m")
	print(f"{output_file_prefix}_train.pt")
	print(f"{output_file_prefix}_test.pt")
	print(f"{output_file_prefix}_time_steps.pt")


if __name__ == "__main__":
    store_periodic_dataset()