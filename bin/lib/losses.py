###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
# 
# Consolidated losses module - contains all loss calculation functions
###########################

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


#####################################################################################################
# Likelihood Functions
#####################################################################################################


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
	# masked_log_lambdas and masked_data 
	n_data_points = masked_data.size()[-1]

	if n_data_points > 0:
		log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
		#log_prob = log_prob / n_data_points
	else:
		log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
	return log_prob


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    """
    Computes the sum of log-likelihoods for a batch of sequences.
    """
    n_data_points = mu_2d.size(-1)
    
    if n_data_points > 0:
        # The Independent wrapper sums the log-probs of the individual Normals.
        gaussian = Independent(Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(data_2d) / (n_data_points)  # Normalize by number of data points
    else:
        # Return a zero tensor with the correct batch dimension
        log_prob = torch.zeros(mu_2d.shape[0]).to(get_device(data_2d))

    return log_prob

def compute_masked_likelihood(mu, data, mask, likelihood_func):
    """
    Computes the likelihood for masked data by summing over the dimensions.
    """
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            dim_log_probs = []
            for j in range(n_dims):
                data_masked = torch.masked_select(data[i,k,:,j], mask[i,k,:,j].bool())
                if data_masked.size(0) == 0:
                    continue  # Skip dimensions with no observations
                
                mu_masked = torch.masked_select(mu[i,k,:,j], mask[i,k,:,j].bool())
                
                log_prob_dim = likelihood_func(mu_masked, data_masked, indices=(i,k,j))
                dim_log_probs.append(log_prob_dim)
            
            if dim_log_probs:
                res.append(torch.stack(dim_log_probs).sum())
            else:
                res.append(torch.tensor(0.0).to(get_device(data)))

    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape((n_traj_samples, n_traj))
    res = res.transpose(0,1)
    return res

def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    """
    Computes the Gaussian log-density for time series, handling masks and multiple samples.
    """
    # Expand data tensor to match samples dimension of mu if needed
    if data.dim() == 3 and mu.dim() == 4:
        if data.size(0) != mu.size(1): # B, T, D vs S, B, T, D
             raise ValueError("Batch dimensions of data and mu do not match.")
        data = data.repeat(mu.size(0), 1, 1, 1)

    if mask is None:
        # Flatten all dimensions except the last one for efficient computation
        n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        
        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        # Use masked computation for datasets with missing values
        func = lambda mu_inner, data_inner, indices: gaussian_log_likelihood(
            mu_inner, data_inner, obsrv_std=obsrv_std, indices=indices
        )
        res = compute_masked_likelihood(mu, data, mask, func)
    
    return res

#####################################################################################################
# Classification Loss Functions
#####################################################################################################

def compute_binary_CE_loss(label_predictions, mortality_label):
	#print("Computing binary classification loss: compute_CE_loss")

	mortality_label = mortality_label.reshape(-1)

	if len(label_predictions.size()) == 1:
		label_predictions = label_predictions.unsqueeze(0)
 
	n_traj_samples = label_predictions.size(0)
	label_predictions = label_predictions.reshape(n_traj_samples, -1)
	
	idx_not_nan = ~torch.isnan(mortality_label)
	if len(idx_not_nan) == 0.:
		print("All are labels are NaNs!")
		ce_loss = torch.Tensor([0.]).to(get_device(mortality_label))
	else:
		if torch.sum(idx_not_nan) == 0.:
			print("All are labels are NaNs!")
			ce_loss = torch.Tensor([0.]).to(get_device(mortality_label))
		else:
			label_predictions = label_predictions[:,idx_not_nan]
			mortality_label = mortality_label[idx_not_nan]

			if len(label_predictions.size()) == 1:
				label_predictions = label_predictions.unsqueeze(0)

			n_traj_samples, n_traj = label_predictions.size()

			label_predictions = label_predictions.reshape(n_traj_samples * n_traj)
			mortality_label = mortality_label.repeat(n_traj_samples)
			
			# Use BCEWithLogitsLoss for binary classification (more numerically stable)
			ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label.float())

	return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask):

	if (len(label_predictions.size()) == 3):
		label_predictions = label_predictions.unsqueeze(0)

	n_traj_samples, n_traj, n_tp, n_dims = label_predictions.size()

	# For each trajectory, we get n_tp predictions and n_tp labels
	# Remove the trajectory axis and take predictions and labels for all time points
	label_predictions = label_predictions.reshape(n_traj_samples * n_traj * n_tp, n_dims)
	true_label = true_label.repeat(n_traj_samples, 1, 1)
	true_label = true_label.reshape(n_traj_samples * n_traj * n_tp, n_dims)

	# choose the maximum
	_, true_label = torch.max(true_label, -1)

	# repeat the mask
	mask = mask.repeat(n_traj_samples, 1, 1)
	mask = mask.reshape(n_traj_samples * n_traj * n_tp)

	if len(mask) == 0:
		ce_loss = torch.Tensor([0.]).to(get_device(label_predictions))
	else:
		ce_loss = nn.CrossEntropyLoss(reduction = 'none')(label_predictions, true_label.long())
		ce_loss = ce_loss * mask
		ce_loss = torch.mean(ce_loss)
	return ce_loss


#####################################################################################################
# MSE Loss Functions
#####################################################################################################

def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse


def compute_mse(mu, data, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		res = compute_masked_likelihood(mu, data, mask, mse)
	return res


#####################################################################################################
# Time Series Specific Loss Functions
#####################################################################################################

def get_derivative_loss(truth, pred_y, time_steps, mask=None, weight=1.0):
	"""Compute derivative matching loss for sharp transitions"""
	# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
	# truth shape  [n_traj, n_tp, n_dim]
	
	# Expand truth to match pred_y's n_traj_samples dimension
	truth_expanded = truth.repeat(pred_y.size(0), 1, 1, 1)

	# Ensure time_steps is 2D, expanding if it's 1D
	if time_steps.dim() == 1:
		# Expand to [batch_size, n_tp] using the batch size from truth
		time_steps = time_steps.unsqueeze(0).expand(truth.shape[0], -1)
	
	# Compute time differences
	dt = time_steps[:, 1:] - time_steps[:, :-1]  # [n_traj, n_tp-1]
	dt = dt.unsqueeze(0).unsqueeze(-1)  # [1, n_traj, n_tp-1, 1]
	dt = dt.repeat(pred_y.size(0), 1, 1, pred_y.size(-1))  # Match pred_y shape
	
	# Compute derivatives using finite differences, preventing division by zero
	truth_deriv = (truth_expanded[:, :, 1:, :] - truth_expanded[:, :, :-1, :]) / (dt + 1e-10)
	pred_deriv = (pred_y[:, :, 1:, :] - pred_y[:, :, :-1, :]) / (dt + 1e-10)
	
	# Apply mask if provided
	if mask is not None:
		mask_deriv = mask.repeat(pred_y.size(0), 1, 1, 1)[:, :, :-1, :]  # Remove last timestep
		truth_deriv = truth_deriv * mask_deriv
		pred_deriv = pred_deriv * mask_deriv
	
	# Compute L2 loss on derivatives
	deriv_diff = (truth_deriv - pred_deriv) ** 2
	deriv_loss = torch.mean(deriv_diff) * weight
	return deriv_loss 


def get_frequency_loss(truth, pred_y, mask=None):
    """
    Encourages frequency content similarity using a log-scale power spectrum loss.
    
    NOTE: This loss is not recommended for variable-length sequences that require masking.
    Masking in the time domain introduces spectral leakage artifacts in the FFT.
    It's best to use this loss on batches where all sequences have the same length.
    """
    # truth shape: [n_traj, n_tp, n_dim]
    # pred_y shape: [n_traj_samples, n_traj, n_tp, n_dim]

    if mask is not None:
        # Warning: Masking before FFT is mathematically problematic.
        # It's better to prepare data in fixed-length batches for this loss.
        print("Warning: Using a mask with get_frequency_loss can introduce FFT artifacts.")
    
    # Reshape for easier processing by collapsing sample and trajectory dimensions
    # New pred_y shape: [batch, n_tp, n_dim] where batch = n_traj_samples * n_traj
    batch_size, n_tp, n_dim = pred_y.shape[1], pred_y.shape[2], pred_y.shape[3]
    pred_y_reshaped = pred_y.reshape(-1, n_tp, n_dim)
    
    # Helper function to compute the log power spectrum
    def compute_log_power_spectrum(signal):
        # signal: [batch, n_tp, n_dim]
        fft = torch.fft.rfft(signal, dim=1, norm='ortho') # Use rfft for real inputs, 'ortho' norm
        power = torch.abs(fft) ** 2
        
        # Add a small epsilon for numerical stability before taking the log
        #log_power = torch.log(power + 1e-8)
        return power
        
    # Compute log power spectra
    pred_log_power = compute_log_power_spectrum(pred_y_reshaped)
    
    # Unsqueeze truth to enable broadcasting, avoiding a memory-intensive .repeat() call
    # truth shape: [1, n_traj, n_tp, n_dim] -> reshaped to [n_traj, n_tp, n_dim]
    truth_reshaped = truth.reshape(batch_size, n_tp, n_dim)
    truth_log_power = compute_log_power_spectrum(truth_reshaped)

    # L2 loss between log power spectra. Broadcasting handles the sample dimension.
    # truth_log_power is broadcast from [n_traj, n_freq, n_dim] to match pred_y's shape.
    freq_loss = torch.mean((pred_log_power - truth_log_power.unsqueeze(0)) ** 2)

    return freq_loss



#####################################################################################################
# Poisson Process Loss Functions
#####################################################################################################

def compute_poisson_proc_likelihood(truth, pred_y, info, mask = None):
	# Compute Poisson likelihood
	# https://math.stackexchange.com/questions/344487/log-likelihood-of-a-realization-of-a-poisson-process
	# Sum log lambdas across all time points
	if mask is None:
		poisson_log_l = torch.sum(info["log_lambda_y"], 2) - info["int_lambda"]
		# Sum over data dims
		poisson_log_l = torch.mean(poisson_log_l, -1)
	else:
		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		mask_repeated = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
		int_lambda = info["int_lambda"]
		f = lambda log_lam, data, indices: poisson_log_likelihood(log_lam, data, indices, int_lambda)
		poisson_log_l = compute_masked_likelihood(info["log_lambda_y"], truth_repeated, mask_repeated, f)
		poisson_log_l = poisson_log_l.permute(1,0)
		# Take mean over n_traj
		#poisson_log_l = torch.mean(poisson_log_l, 1)
		
	# poisson_log_l shape: [n_traj_samples, n_traj]
	return poisson_log_l

