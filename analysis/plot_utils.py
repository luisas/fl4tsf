
import os
from torch.utils.data import DataLoader
from lib import utils
from lib.latent_ode import LatentODE
from lib.plot import plot_trajectories
import matplotlib.pyplot as plt
import torch
from flower.get_dataset import basic_collate_fn
import pandas as pd
import json


def plot_n_outputs(model, dataset, timestamps, n = 3, id = 10):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # prepare data loader 
    batch_size = 16
    dataset_name = "periodic"
    sample_tp = 0.5
    cut_tp = None
    extrap = False
    experimentID = id

    testloader = DataLoader(dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, timestamps, dataset_name, sample_tp, cut_tp, extrap, data_type = "test"))
    testloader = utils.inf_generator(testloader)
    test_dict = utils.get_next_batch(testloader)

    # prepare dictionary in the format needed for prediction and plotting
    data =  test_dict["data_to_predict"]
    time_steps = test_dict["tp_to_predict"]
    mask = test_dict["mask_predicted_data"]

    observed_data =  test_dict["observed_data"]
    observed_time_steps = test_dict["observed_tp"]
    observed_mask = test_dict["observed_mask"]

    n_traj_to_show = n

    ax_traj = []
    fig, ax_traj = plt.subplots(1, n_traj_to_show)
    fig.set_size_inches(15, 3)

    time_steps_to_predict = time_steps
    if isinstance(model, LatentODE):
        # sample at the original time points
        time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100).to(device)

    with torch.no_grad():

        # Here use the model to obtain the predictions
        reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
            observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 10)

        # plot only 10 trajectories
        data_for_plotting = observed_data[:n_traj_to_show]
        mask_for_plotting = observed_mask[:n_traj_to_show]

        data_full = data[:n_traj_to_show]

        reconstructions_for_plotting = reconstructions.mean(dim=0)[:n_traj_to_show]
        reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

        dim_to_show = 0
        max_y = max(
            data_for_plotting[:,:,dim_to_show].cpu().numpy().max(),
            reconstructions[:,:,dim_to_show].cpu().numpy().max())
        min_y = min(
            data_for_plotting[:,:,dim_to_show].cpu().numpy().min(),
            reconstructions[:,:,dim_to_show].cpu().numpy().min())

        ############################################
        # Plot reconstructions, true postrior and approximate posterior

        cmap = plt.colormaps['Set1']
        for traj_id in range(n_traj_to_show):

            # Plot observations
            plot_trajectories(ax_traj[traj_id], 
                data_full[traj_id].unsqueeze(0), observed_time_steps, 
                #mask = mask_for_plotting[traj_id].unsqueeze(0),
                min_y = min_y, max_y = max_y, #title="True trajectories", 
                marker = 'o', linestyle='', dim_to_show = dim_to_show, markersize= 2, alpha =0.9,
                color = "grey")

            # Plot observations
            plot_trajectories(ax_traj[traj_id], 
                data_for_plotting[traj_id].unsqueeze(0), observed_time_steps, 
                mask = mask_for_plotting[traj_id].unsqueeze(0),
                min_y = min_y, max_y = max_y, #title="True trajectories", 
                marker = 'o', linestyle='', dim_to_show = dim_to_show, add_to_plot= True, markersize= 2.5, alpha =0.5,
                color = "black")

            # Plot reconstructions
            plot_trajectories(ax_traj[traj_id],
                reconstructions_for_plotting[traj_id].unsqueeze(0), time_steps_to_predict, 
                min_y = min_y, max_y = max_y, title="Sample {}".format(traj_id), dim_to_show = dim_to_show,
                add_to_plot = True, marker = '', color =  "darkorange", linewidth = 1.3, linestyle='-', alpha = 0.9)
            # Plot std
            # plot_std(ax_traj[traj_id], 
			# 	reconstructions_for_plotting[traj_id].unsqueeze(0), reconstr_std[traj_id].unsqueeze(0), 
			# 	time_steps_to_predict, alpha=0.5, color = "grey", add_to_plot= True)

            # y_padding = 0.1 * (max_y - min_y)
            # min_y -= y_padding
            # max_y += y_padding
            ax_traj[traj_id].set_ylim(min_y - 0.3, max_y+ 0.3)
            # print(min_y)
            
        plt.show()
    


def read_loss_file(file):
    # Read the meta.csv file
    meta_file = file.replace("results.json", "meta.csv")
    meta_data = pd.read_csv(meta_file)
    lr = meta_data['lr'].item()
    batch_size = meta_data['batch_size'].item()
    clipping = meta_data['gradientclipping'].item()
    lrdecay = meta_data['lrdecay'].item()
    nlocalepochs = meta_data['localepochs'].item()
    
    # Read the results.json file
    with open(file, 'r') as f:
        data = json.load(f)
    
    # Plot centralized evaluate
    df_centralized_evaluate = pd.DataFrame(data['centralized_evaluate'])
    df_federated_evaluate = pd.DataFrame(data['federated_evaluate'])
    df_aggregation = pd.DataFrame(data['aggregation'])
    # Add the learning rate to the DataFrame
    df_federated_evaluate['lr'] = lr
    df_centralized_evaluate['lr'] = lr
    # Add the batch size to the DataFrame
    df_federated_evaluate['batch_size'] = batch_size
    df_centralized_evaluate['batch_size'] = batch_size
    # Add the clipping to the DataFrame
    df_federated_evaluate['clipping'] = clipping
    df_centralized_evaluate['clipping'] = clipping
    # add decay
    df_federated_evaluate['lrdecay'] = lrdecay
    df_centralized_evaluate['lrdecay'] = lrdecay
    # add nlocalepochs
    df_federated_evaluate['localepochs'] = nlocalepochs
    df_centralized_evaluate['localepochs'] = nlocalepochs
    # add decay onset 
    df_federated_evaluate['decay_onset'] = meta_data['decay_onset'].item()
    df_centralized_evaluate['decay_onset'] = meta_data['decay_onset'].item()
    # add datasetname
    df_federated_evaluate['dataset_name'] = meta_data['dataset_name'].item()
    df_centralized_evaluate['dataset_name'] = meta_data['dataset_name'].item()
    # TODO clean 
    # extrat the last part of the dataset name
    df_federated_evaluate['offset'] = df_federated_evaluate['dataset_name'].apply(lambda x: x.split('_')[-1])
    df_centralized_evaluate['offset'] = df_centralized_evaluate['dataset_name'].apply(lambda x: x.split('_')[-1])
    # make it float
    df_federated_evaluate['offset'] = df_federated_evaluate['offset'].astype(float)
    df_centralized_evaluate['offset'] = df_centralized_evaluate['offset'].astype(float)

    # alpha
    df_federated_evaluate['alpha'] = meta_data['alpha'].item()
    df_centralized_evaluate['alpha'] = meta_data['alpha'].item()

    # Aggregation column 
    # Check if file contins "FedAvg" in the path name
    if "FedAvg" in file:
        # if the path name contains "FedAvg" then add aggregation column "FedAvg"
        df_federated_evaluate['aggregation'] = "FedAvg"
        df_centralized_evaluate['aggregation'] = "FedAvg"
    else:
        # if the path name does not contain "FedAvg" then add aggregation column "FedProx"
        df_federated_evaluate['aggregation'] = "FedODE"
        df_centralized_evaluate['aggregation'] = "FedODE"


    # if the path name contains "FedAvg" then add aggregation column "FedAvg"
    df_federated_evaluate['hyperparameters'] = df_federated_evaluate.apply(lambda x: f"lr: {x['lr']}, batch_size: {x['batch_size']}, clipping: {x['clipping']}, lrdecay: {x['lrdecay']}, nlocalepochs: {x['localepochs']}", axis=1)
    df_centralized_evaluate['hyperparameters'] = df_centralized_evaluate.apply(lambda x: f"lr: {x['lr']}, batch_size: {x['batch_size']}, clipping: {x['clipping']}, lrdecay: {x['lrdecay']}, nlocalepochs: {x['localepochs']}", axis=1)
    # combine lr, batch size and clipping into one column

    df_federated_evaluate['type'] = "federated"
    df_centralized_evaluate['type'] = "centralized"

    # modify centralized_loss into loss
    df_centralized_evaluate.rename(columns={'centralized_loss': 'loss'}, inplace=True)
    # modify federated_loss into loss
    df_federated_evaluate.rename(columns={'federated_evaluate_loss': 'loss'}, inplace=True)

    return df_centralized_evaluate, df_federated_evaluate, df_aggregation


def prepare_df_for_plotting(result_json, prefix = "noise", convergence_range=1.1):
    df_summary_centralized = pd.DataFrame()
    df_summary_federated = pd.DataFrame()
    df_summary_aggregation = pd.DataFrame()
    for file in result_json:
        noise = float(file.split(f"{prefix}_")[-1].split("_")[0].split("/")[0])
        df_centralized_evaluate, df_federated_evaluate, df_aggregation = read_loss_file(file)
        df_centralized_evaluate[prefix] = noise
        df_summary_centralized = pd.concat([df_summary_centralized, df_centralized_evaluate], ignore_index=True)
        df_federated_evaluate[prefix] = noise
        df_summary_federated = pd.concat([df_summary_federated, df_federated_evaluate], ignore_index=True)
        df_aggregation[prefix] = noise
        df_summary_aggregation = pd.concat([df_summary_aggregation, df_aggregation], ignore_index=True)

    # find the min loss for each noise, aggreagation, type 
    df_summary_centralized['min_loss'] = df_summary_centralized.groupby([prefix, 'aggregation', 'type'])['loss'].transform('min')

    # per noise, aggregation, type get the convergence_round, which is the min round where the loss is within 1/10 of the min loss
    df_summary_centralized['convergence_round'] = df_summary_centralized.groupby([prefix, 'aggregation', 'type'])['round'].transform(
        lambda x: x[df_summary_centralized['loss'] <= df_summary_centralized['min_loss'] * convergence_range].min()
    )

    df_summary_federated['min_loss'] = df_summary_federated.groupby([prefix, 'aggregation', 'type'])['loss'].transform('min')
    df_summary_federated['convergence_round'] = df_summary_federated.groupby([prefix, 'aggregation', 'type'])['round'].transform(
        lambda x: x[df_summary_federated['loss'] <= df_summary_federated['min_loss'] * convergence_range].min()
    )

    df_summary_federated = df_summary_federated[['round', 'loss', 'aggregation', 'type', 'alpha', prefix, 'min_loss', 'convergence_round']]
    df_summary_centralized = df_summary_centralized[['round', 'loss', 'aggregation', 'type', 'alpha', prefix, 'min_loss', 'convergence_round']]
    df_summary = pd.concat([df_summary_federated, df_summary_centralized], ignore_index=True)
    # df_summary = df_summary[['aggregation', 'type', 'alpha', prefix, 'min_loss', 'convergence_round']]
    # df_summary = df_summary.drop_duplicates()
    # extarct n_steps_list[0] into c0_steps and n_steps_list[1] into c1_steps
    df_summary_aggregation["steps_0"] = df_summary_aggregation['num_steps_list'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df_summary_aggregation["steps_1"] = df_summary_aggregation['num_steps_list'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
    # remove nan 
    df_summary_aggregation = df_summary_aggregation.dropna(subset=['steps_0', 'steps_1'])
    # do same thing for lambda
    df_summary_aggregation["lambda_0"] = df_summary_aggregation['lambdas'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df_summary_aggregation["lambda_1"] = df_summary_aggregation['lambdas'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
    df_agg = df_summary_aggregation[["round", "steps_0", "steps_1", "lambda_0", "lambda_1", prefix, "alpha"]]

    df0 = df_agg[['round', prefix, 'lambda_0', 'steps_0',  'alpha']].copy()
    df0['client'] = 0
    df0 = df0.rename(columns={'lambda_0': 'lambda', 'steps_0': 'steps'})
    df1 = df_agg[['round', prefix, 'lambda_1', 'steps_1',  'alpha']].copy()
    df1.rename(columns={'lambda_1': 'lambda', 'steps_1': "steps"}, inplace=True)
    df1['client'] = 1
    df_summary_lambdas = pd.concat([df0, df1], ignore_index=True)
    return df_summary, df_summary_lambdas