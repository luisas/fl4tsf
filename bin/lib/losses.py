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

def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
	n_data_points = mu_2d.size()[-1]

	if n_data_points > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
		log_prob = gaussian.log_prob(data_2d) 
		log_prob = log_prob / n_data_points 
	else:
		log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
	return log_prob


def poisson_log_likelihood(masked_log_lambdas, masked_data, indices, int_lambdas):
	# masked_log_lambdas and masked_data 
	n_data_points = masked_data.size()[-1]

	if n_data_points > 0:
		log_prob = torch.sum(masked_log_lambdas) - int_lambdas[indices]
		#log_prob = log_prob / n_data_points
	else:
		log_prob = torch.zeros([1]).to(get_device(masked_data)).squeeze()
	return log_prob


def masked_gaussian_log_density(mu, data, obsrv_std, mask = None):
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
	
		res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
		res = res.reshape(n_traj_samples, n_traj).transpose(0,1)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std, indices = indices)
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
	return deriv_loss / 500


def get_frequency_loss(truth, pred_y, mask=None):
	"""Encourage frequency content similarity (penalize constant predictions)"""
	# truth shape: [n_traj, n_tp, n_dim]
	# pred_y shape: [n_traj_samples, n_traj, n_tp, n_dim]
	
	truth_expanded = truth.repeat(pred_y.size(0), 1, 1, 1)
	
	if mask is not None:
		mask_expanded = mask.repeat(pred_y.size(0), 1, 1, 1)
		truth_expanded = truth_expanded * mask_expanded
		pred_y = pred_y * mask_expanded
	
	# Compute FFT for each trajectory and dimension
	def compute_power_spectrum(signal):
		# signal: [batch, n_tp, n_dim]
		fft = torch.fft.fft(signal, dim=1)
		power = torch.abs(fft) ** 2
		return power[:, :signal.size(1)//2, :]  # Keep only positive frequencies
	
	truth_power = compute_power_spectrum(truth_expanded)
	pred_power = compute_power_spectrum(pred_y)
	
	# L2 loss between power spectra
	freq_loss = torch.mean((truth_power - pred_power) ** 2)
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


#####################################################################################################
# Helper Functions
#####################################################################################################

def compute_masked_likelihood(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	res = []
	for i in range(n_traj_samples):
		for k in range(n_traj):
			for j in range(n_dims):

				data_masked = torch.masked_select(data[i,k,:,j], mask[i,k,:,j].bool())

				# Skip dimensions with no observations
				if (torch.sum(mask[i,k,:,j]) == 0):
					continue
				mu_masked = torch.masked_select(mu[i,k,:,j], mask[i,k,:,j].bool())
				log_prob = likelihood_func(mu_masked, data_masked, indices = (i,k,j))
				res.append(log_prob)
	# shape: [n_traj*n_traj_samples, 1]

	res = torch.stack(res, 0).to(get_device(data))
	res = res.reshape((n_traj_samples, n_traj, n_dims))
	# Take mean over the number of dimensions
	res = torch.mean(res, -1) # !!!!!!!!!!! changed from sum to mean
	res = res.transpose(0,1)
	return res