###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.latent_ode import LatentODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver

from torch.distributions.normal import Normal
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson

#####################################################################################################

def create_LatentODE_model(latents, poisson, units, gen_layers, rec_dims, rec_layers, 
                           gru_units, z0_encoder, classif, linear_classif, train_classif_w_reconstr, 
                           input_dim, z0_prior, obsrv_std, device, 
                           classif_per_tp=False, n_labels=1, 
                           solver_method='dopri5', encoder_solver_method='euler',
                           use_jit=True):  # Updated defaults
    
    dim = latents
    if poisson:
        lambda_net = utils.create_net(dim, input_dim, 
            n_layers=1, n_units=units, nonlinear=nn.Tanh)

        # ODE function produces the gradient for latent state and for poisson rate
        ode_func_net = utils.create_net(dim * 2, latents * 2, 
            n_layers=gen_layers, n_units=units, nonlinear=nn.Tanh)

        gen_ode_func = ODEFunc_w_Poisson(
            input_dim=input_dim, 
            latent_dim=latents * 2,
            ode_func_net=ode_func_net,
            lambda_net=lambda_net,
            device=device).to(device)
    else:
        dim = latents 
        ode_func_net = utils.create_net(dim, latents, 
            n_layers=gen_layers, n_units=units, nonlinear=nn.Tanh)
        
        gen_ode_func = ODEFunc(
            input_dim=input_dim, 
            latent_dim=latents, 
            ode_func_net=ode_func_net,
            device=device).to(device)

    z0_diffeq_solver = None
    n_rec_dims = rec_dims
    enc_input_dim = int(input_dim) * 2  # we concatenate the mask
    gen_data_dim = input_dim

    z0_dim = latents
    if poisson:
        z0_dim += latents  # predict the initial poisson rate

    if z0_encoder == "odernn":
        ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
            n_layers=rec_layers, n_units=units, nonlinear=nn.Tanh)

        rec_ode_func = ODEFunc(
            input_dim=enc_input_dim, 
            latent_dim=n_rec_dims,
            ode_func_net=ode_func_net,
            device=device).to(device)

        # Use simpler solver for encoder, stiff solver for main dynamics
        z0_diffeq_solver = DiffeqSolver(
            enc_input_dim, rec_ode_func, encoder_solver_method, latents, 
            odeint_rtol=1e-5, odeint_atol=1e-5, device=device)
            
        encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
            z0_dim=z0_dim, n_gru_units=gru_units, device=device).to(device)

    elif z0_encoder == "rnn":
        encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
            lstm_output_size=n_rec_dims, device=device).to(device)
    else:
        raise Exception("Unknown encoder for Latent ODE model: " + z0_encoder)

    decoder = Decoder(latents, gen_data_dim, n_units=units).to(device)

    # Use Tsit5 adaptive solver for main dynamics (handles stiffness well)
    diffeq_solver = DiffeqSolver(
        gen_data_dim, gen_ode_func, solver_method, latents, 
        odeint_rtol=1e-5, odeint_atol=1e-4, device=device)  # Tighter tolerances

    model = LatentODE(
        input_dim=gen_data_dim, 
        latent_dim=latents, 
        encoder_z0=encoder_z0, 
        decoder=decoder, 
        diffeq_solver=diffeq_solver, 
        z0_prior=z0_prior, 
        device=device,
        obsrv_std=obsrv_std,
        use_poisson_proc=poisson, 
        use_binary_classif=classif,
        linear_classifier=linear_classif,
        classif_per_tp=classif_per_tp,
        n_labels=n_labels,
        train_classif_w_reconstr=train_classif_w_reconstr,
        use_jit=use_jit  # Pass the JIT flag
    ).to(device)

    return model