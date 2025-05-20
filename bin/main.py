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


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU


def main():
    
    print("Starting Flower server...")

    cpus_for_this_ray_instance = 2

    # Check RAY_TMPDIR environment variable
    ray_tmpdir = os.environ.get("RAY_TMPDIR", "Not set")
    ray_socket_dir = os.environ.get("RAY_SOCKET_DIR", os.path.join(ray_tmpdir, "s") if ray_tmpdir != "Not set" else None)
    
    # Initialize Ray with explicit socket paths
    ray.init(
        logging_level=logging.INFO,
        include_dashboard=False,
        num_cpus=cpus_for_this_ray_instance,
        num_gpus=1,
        _plasma_directory=ray_socket_dir,  # Use the very short socket directory
        _temp_dir=ray_tmpdir,  # Use the main temporary directory
        _enable_object_reconstruction=True,
    )

    client = ClientApp(client_fn)
    server = ServerApp(server_fn=server_fn )

    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1}}

    run_simulation(
        server_app=server,
        client_app=client,
        backend_config=backend_config,
        num_supernodes = 2,
    )
            
    ray.shutdown()




if __name__ == "__main__": 
    print("Starting the script ...")
    main()