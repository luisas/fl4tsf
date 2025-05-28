import argparse
import logging
import ray
import sys
import decimal
import re

from filelock import FileLock
import json
import os
import torch 
from flower.server_app import ServerApp, server_fn
from flower.client_app import ClientApp, client_fn
from flwr.simulation import run_simulation
from flwr.common import Config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(ncpus = 0, ngpus=0, raydir = None, ray_socket_dir=None, nclients = 2):
    
    print("Starting Flower server...")
    # print the device 
    print(f"Using device: {DEVICE}")
    
    # Initialize Ray with explicit socket paths
    ray.init(
        logging_level=logging.INFO,
        include_dashboard=False,
        num_cpus=ncpus,
        num_gpus=ngpus,
        _plasma_directory=ray_socket_dir,  # Use the very short socket directory
        _temp_dir=raydir,  # Use the main temporary directory
        _enable_object_reconstruction=True,
    )

    client = ClientApp(client_fn)
    server = ServerApp(server_fn=server_fn )

    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": { "num_gpus": 1, "num_cpus":1 }}

    run_simulation(
        server_app=server,
        client_app=client,
        backend_config=backend_config,
        num_supernodes =nclients,
    )
            
    ray.shutdown()




if __name__ == "__main__": 
    print("Starting the script ...")
    # parser for ncpus
    parser = argparse.ArgumentParser(description="Flower server")
    parser.add_argument(
        "--ncpus",
        type=int,
        default=2,
        help="Number of CPUs to use for the server",
    )
    parser.add_argument(
        "--raydir",
        type=str,
        default=None,
        help="Directory for Ray temporary files",
    )
    parser.add_argument(
        "--ray_socket_dir",
        type=str,
        default=None,
        help="Directory for Ray socket files",
    )

    parser.add_argument(
        "--ngpus",
        type=int,
        default=0,
        help="Number of GPUs available"
    )

    parser.add_argument(
        "--nclients",
        type=int,
        default=2,
        help="Number of clients to simulate"
    )

    args = parser.parse_args()
    main(args.ncpus, args.ngpus, args.raydir, args.ray_socket_dir, args.nclients)