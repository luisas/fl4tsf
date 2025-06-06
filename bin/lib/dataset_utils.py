###############################################################################
# Get amplitudes and frequencies according to drichlet distribution
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from lib.generate_timeseries import Periodic_1d
import torch
from collections import Counter
from matplotlib import cm
from torch.distributions import uniform
import pandas as pd
import os
import seaborn as sns
import random

def create_meta(num_clients, base_amps, base_freqs, alpha, n = 100):
    """
    Create a metadata dictionary for clients with their respective
    amplitudes and frequencies.
    """
    alpha_vector_amps = [alpha] * len(base_amps)
    alpha_vector_freqs = [alpha] * len(base_freqs)
    meta = pd.DataFrame()
    for i in range(num_clients):
        param_dist_amp = np.random.dirichlet(alpha_vector_amps)
        param_dist_freq = np.random.dirichlet(alpha_vector_freqs)
        
        amps = np.random.choice(base_amps, p=param_dist_amp, size=n)
        freqs = np.random.choice(base_freqs, p=param_dist_freq, size=n)
        
        df = pd.DataFrame({
            'client_id': i,
            'amplitude': amps,
            'frequency': freqs,
            "alpha": alpha
        })
        meta = pd.concat([meta, df], ignore_index=True)
        
    return meta

#########################################
# Plot partitioning clients
#########################################


# Function to plot for each alpha
def plot_stacked(data, **kwargs):
    ax = plt.gca()
    clients = data['client_id'].unique()
    x = np.arange(len(clients))
    width = 0.4

    amp_counts = data.groupby(['client_id', 'amplitude']).size().unstack(fill_value=0).reindex(clients)
    freq_counts = data.groupby(['client_id', 'frequency']).size().unstack(fill_value=0).reindex(clients)

    amp_vals = amp_counts.columns.tolist()
    freq_vals = freq_counts.columns.tolist()
    colors_amp = [cm.Reds((i + 2) / (len(amp_vals) + 2)) for i in range(len(amp_vals))]
    colors_freq = [cm.Blues((i + 2) / (len(freq_vals) + 2)) for i in range(len(freq_vals))]

    bottom = np.zeros(len(clients))
    for j, val in enumerate(amp_vals):
        ax.bar(x - width/2, amp_counts[val], width, bottom=bottom,
               label=f"Amp {val}", color=colors_amp[j])
        bottom += amp_counts[val]

    bottom = np.zeros(len(clients))
    for j, val in enumerate(freq_vals):
        ax.bar(x + width/2, freq_counts[val], width, bottom=bottom,
               label=f"Freq {val}", color=colors_freq[j])
        bottom += freq_counts[val]

    ax.set_xticks(x)
    client_labels = [f"client {i}" for i in clients]
    ax.set_xticklabels(client_labels, rotation=45)
    # only set y label for the first plot
    ax.set_ylabel("# timeseries")
    # only set y label for the first plot
    


def sample_timesteps(n_total_tp, max_t_extrap, distribution_type='uniform'):
    #time_steps = torch.linspace(0, max_t_extrap, n_total_tp)
    n_total_tp = n_total_tp - 1  # Adjust for the initial time step at 0.0
    if n_total_tp <= 0:
        raise ValueError("n_total_tp must be greater than 1 to sample time steps.")
    if distribution_type == 'uniform':
        # Sample uniformly distributed time steps
        distribution = uniform.Uniform(torch.Tensor([0.0]), torch.Tensor([max_t_extrap]))
        time_steps = distribution.sample(torch.Size([n_total_tp]))[:, 0]
    elif distribution_type == 'exponential':
        distribution = torch.distributions.Exponential(3.0)
        time_steps = distribution.sample((n_total_tp,))
        time_steps = time_steps / time_steps.max() * max_t_extrap
    elif distribution_type == 'lognormal':
        mean = random.uniform(0.0, 1.0)
        std = random.uniform(0.1, 2.0)
        distribution = torch.distributions.LogNormal(mean, std)
        time_steps = distribution.sample((n_total_tp,))
        time_steps = time_steps / time_steps.max() * max_t_extrap
    elif distribution_type == 'beta':
        left = 1.0
        right = 1.0
        a = random.uniform(left,right)
        b = random.uniform(left,right)
        distribution = torch.distributions.Beta(a, b)
        time_steps = distribution.sample((n_total_tp,))
        time_steps = time_steps * max_t_extrap

    time_steps = torch.cat((torch.Tensor([0.0]), time_steps))
    time_steps = torch.sort(time_steps)[0]
    return time_steps


def get_dataset(amplitude, frequency, timesteps, n_samples=50, noise_weight=0, distribution='uniform', n_total_tp=None, max_t=None):
    """
    Generate a dataset of periodic functions with given amplitude and frequency.
    """
    # Create time steps for extrapolation

    dataset_obj = None
    ##################################################################
    # Sample a periodic function

    # check if amplitude is a single value
    if isinstance(amplitude, (int, float)):
        init_amplitude = amplitude
        final_amplitude = amplitude
    elif isinstance(amplitude, (list, np.ndarray)):
        if len(amplitude) == 2:
            init_amplitude, final_amplitude = amplitude
        else:
            raise ValueError("Amplitude must be a single value or a list of two values [init_amplitude, final_amplitude].")
    
    if isinstance(frequency, (int, float)):
        init_freq = frequency
        final_freq = frequency
    elif isinstance(frequency, (list, np.ndarray)):
        if len(frequency) == 2:
            init_freq, final_freq = frequency
        else:
            raise ValueError("Frequency must be a single value or a list of two values [init_freq, final_freq].")

    dataset_obj = Periodic_1d(
        init_freq = init_freq, init_amplitude = init_amplitude,
        final_amplitude = final_amplitude, final_freq = final_freq, 
        z0 = 1.)

    ##################################################################
    

    # if timesteps is None and max_t is not None and n_total_tp is not None:
    dataset = None
    if timesteps is None and max_t is not None and n_total_tp is not None:
        print(f"Sampling {n_samples} samples with {n_total_tp} time points and max_t={max_t}")
        dataset = []
        all_timesteps = []
        for i in range(n_samples):
            timesteps = sample_timesteps(n_total_tp, max_t, distribution_type=distribution)
            traj = dataset_obj.sample_traj(timesteps, n_samples=1, noise_weight=noise_weight)
            dataset.append(traj)
            all_timesteps.append(timesteps)
        dataset = torch.cat(dataset, dim=0)
        all_timesteps = torch.stack(all_timesteps, dim=0)
    elif timesteps is not None and max_t is None and n_total_tp is None:
        print(f"Sampling {n_samples} samples with provided timesteps")
        dataset = dataset_obj.sample_traj(timesteps, n_samples = n_samples, noise_weight = noise_weight)
        all_timesteps = timesteps.unsqueeze(0).repeat(n_samples, 1)
    else:
        raise ValueError("Either provide timesteps or n_total_tp and max_t. Not both.")

    return dataset, all_timesteps


#########################################
# Create combinations of datasets
#########################################

def create_dataset_with_combinations(combinations, time_steps, n_samples_per_combination, noise_weight = 0, distribution='uniform', n_total_tp=None, max_t_extrap=None):
    """
    Create datasets for each combination of amplitude and frequency.
    """

    datasets = []
    for amp, freq in combinations:
        dataset, time_steps_extrap = get_dataset(amp, freq, time_steps, noise_weight=noise_weight, n_samples=n_samples_per_combination, distribution = distribution, n_total_tp=n_total_tp, max_t=max_t_extrap)
        # make amp same size as dataset
        # no tensor 
        amp = np.full((dataset.shape[0],), amp)
        freq = np.full((dataset.shape[0],), freq)
        datasets.append((amp, freq, dataset, time_steps_extrap))
    return datasets


#########################################
# Merge all datasets for centralized training
#########################################
def merge_datasets(datasets):
    """
    Merge all datasets into one tensor.
    """
    merged_data = []
    amps = []
    freqs = []
    time_steps = []
    for amp, freq, dataset, time_steps_extrap in datasets:
        merged_data.append(dataset)
        # append content of amp and freq to the list in numpy 
        amps.extend(amp)  # reshape to column vector
        freqs.extend(freq)
        time_steps.append(time_steps_extrap)

    # Concatenate along the first dimension (samples)
    merged_data = torch.cat(merged_data, dim=0)
    time_steps_merged = torch.cat(time_steps, dim=0)
    return amps, freqs, merged_data, time_steps_merged


#########################################
# Store
#########################################
def store_dataset(train, test, dataset_prefix, path_prefix, client_prefix = None):
    """
    Store the dataset to disk.
    """
    if client_prefix is not None:
        output_file_prefix = f"{path_prefix}/{dataset_prefix}/{client_prefix}"
    else:
        output_file_prefix = f"{path_prefix}/{dataset_prefix}/{dataset_prefix}"
    # if it doesn't exist, create the directory
    if not os.path.exists(f"{path_prefix}/{dataset_prefix}"):
        os.makedirs(f"{path_prefix}/{dataset_prefix}", exist_ok=True)

    # save the dataset to a file
    with open(f"{output_file_prefix}_train.pt", "wb") as f:
        torch.save(train["data"], f)
    with open(f"{output_file_prefix}_test.pt", "wb") as f:
        torch.save(test['data'], f)
    with open(f"{output_file_prefix}_train_amps.pt", "wb") as f:
        torch.save(train['amplitude'], f)
    with open(f"{output_file_prefix}_train_freqs.pt", "wb") as f:
        torch.save(train['frequency'], f)
    with open(f"{output_file_prefix}_test_amps.pt", "wb") as f:
        torch.save(test['amplitude'], f)
    with open(f"{output_file_prefix}_test_freqs.pt", "wb") as f:
        torch.save(test['frequency'], f)
    with open(f"{output_file_prefix}_time_steps_train.pt", "wb") as f:
        torch.save(train['time_steps'], f)
    with open(f"{output_file_prefix}_time_steps_test.pt", "wb") as f:
        torch.save(test['time_steps'], f)
    with open(f"{output_file_prefix}_time_steps.pt", "wb") as f:
         torch.save(train['time_steps'][0], f)
    
    # print message with colors
    print("\033[92mPeriodic dataset created and saved to disk.\033[0m")
    print(f"{output_file_prefix}_train.pt")
    print(f"{output_file_prefix}_test.pt")
    print(f"{output_file_prefix}_time_steps.pt")


def split_train_test_with_meta(full_dataset, train_fraq = 0.8):
	amps, freqs, data, time_steps = full_dataset

	# shuffle the data
	indices = torch.randperm(data.size(0))
	data = data[indices]
	amps = torch.tensor(amps)[indices]
	freqs = torch.tensor(freqs)[indices]
	time_steps = time_steps[indices]
	# time_steps is already sorted, so we don't shuffle it

	n_samples = data.size(0)
	# amps
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]
	amps_train = amps[:int(n_samples * train_fraq)]
	freqs_train = freqs[:int(n_samples * train_fraq)]
	amps_test = amps[int(n_samples * train_fraq):]
	freqs_test = freqs[int(n_samples * train_fraq):]
	time_steps_train = time_steps[:int(n_samples * train_fraq)]
	time_steps_test = time_steps[int(n_samples * train_fraq):]
	train = {
		'data': data_train,
		'amplitude': amps_train,
		'frequency': freqs_train,
		'time_steps': time_steps_train
	}
	test = {
		'data': data_test,
		'amplitude': amps_test,
		'frequency': freqs_test,
		'time_steps': time_steps_test
	}
	return train, test

#########################################
# plot one sample from each combination
#########################################
def plot_all_combinations(datasets, scatter = False):
    """
    Plot one sample from each combination of amplitude and frequency.
    """
    plt.figure(figsize=(12, 8))
    for i, (amp, freq, dataset, time_steps_extrap) in enumerate(datasets):
        y = dataset[0, :, :].squeeze().cpu().numpy()  # take the first sample
        x = time_steps_extrap[0, :].cpu().numpy()  # take the first time steps
        plt.subplot(4, 4, i + 1)
        if scatter:
            plt.scatter(x, y, alpha=0.5, marker='o', s=2)
        else:
            plt.plot(x, y)
        plt.title(f"Amp: {amp[0]}, Freq: {freq[0]}")
        plt.xlabel('Time')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

# plot distribution of amplitudes and frequencies in train and test set
def plot_distribution(amps_train, freqs_train, amps_test, freqs_test):
    """
    Plot the distribution of amplitudes and frequencies in train and test set.
    """
    plt.figure(figsize=(7, 4))
    
    # Amplitudes
    plt.subplot(1, 2, 1)
    sns.kdeplot(amps_train, color='blue', label='Train', fill =True, alpha=0.5)
    sns.kdeplot(amps_test, color='orange', label='Test', fill =True, alpha=0.5)
    plt.xlabel('amplitude')
    plt.ylabel('density')

    
    # Frequencies
    plt.subplot(1, 2, 2)
    sns.kdeplot(freqs_train, color='blue', label='Train', fill =True, alpha=0.5)
    sns.kdeplot(freqs_test, color='orange', label='Test', fill =True, alpha=0.5)
    plt.xlabel('frequency')
    plt.ylabel('density')
    plt.legend()
    # legend outside of plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()