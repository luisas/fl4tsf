
import os

from torch.utils.data import DataLoader
from flower.get_dataset import get_dataset, basic_collate_fn
from lib import utils
from lib.latent_ode import LatentODE
from lib.plot import plot_trajectories
import matplotlib.pyplot as plt
import torch
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