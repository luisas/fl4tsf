import argparse
import logging
import ray
import sys
import decimal
import re


from fl_flwr_Experiment import FLWRExperiment
from filelock import FileLock
import json
import os
from fl_flwr_Results_Manager import config_structure_list, configs_dict_possibilities, ROOT_DIR  # Ensure this list is accurate
import torch.multiprocessing as mp

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Handler for ALL levels to stdout (SLURM output file)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)  # Capture all levels
stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

# Handler for WARNING and above to stderr (SLURM error file)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(level=logging.DEBUG, handlers=[stdout_handler, stderr_handler])

log = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="Run a single Flower Experiment via SLURM")

    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=configs_dict_possibilities["dataset_name"],
        help="Select the dataset for the experiment."
    )
    parser.add_argument(
        "--partitioner_name",
        required=True,
        choices=configs_dict_possibilities["partitioner_name"],
        help="Select the partitioner."
    )
    parser.add_argument(
        "--num_partitions",
        required=True,
        type=int,
        help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner_parameter",
        required=True,
        help="Partitioner parameter."
    )
    parser.add_argument(
        "--strategy_name",
        required=True,
        choices=configs_dict_possibilities["strategy_name"],
        help="Select the strategy."
    )
    parser.add_argument(
        "--optimizer_name",
        required=True,
        choices=configs_dict_possibilities["optimizer_name"],
        help="Select the optimizer."
    )
    parser.add_argument(
        "--criterion_name",
        required=True,
        choices=configs_dict_possibilities["criterion_name"],
        help="Select the criterion."
    )
    parser.add_argument(
        "--normalization",
        required=True,
        choices=configs_dict_possibilities["normalization"],
        help="Select the normalization."
    )
    parser.add_argument(
        "--batch_size_type",
        required=True,
        choices=configs_dict_possibilities["batch_size_type"],
        help="Select the batch size type."
    )
    parser.add_argument(
        "--batch_size_config",
        required=True,
        help="Batch size configuration."
    )
    parser.add_argument(
        "--learning_rate_type",
        required=True,
        choices=configs_dict_possibilities["learning_rate_type"],
        help="Select the learning rate type."
    )
    parser.add_argument(
        "--learning_rate_config",
        required=True,
        type=float,
        help="Learning rate configuration."
    )
    parser.add_argument(
        "--num_rounds",
        required=True,
        type=int,
        help="Number of rounds."
    )
    parser.add_argument(
        "--local_epochs_type",
        required=True,
        choices=configs_dict_possibilities["local_epochs_type"],
        help="Select the local epochs type."
    )
    parser.add_argument(
        "--local_epochs_config",
        required=True,
        type=int,
        help="Local epochs configuration."
    )
    parser.add_argument(
        "--save_parameters",
        required=True,
        help="Whether to save parameters."
    )
    parser.add_argument(
        "--save_probs",
        required=True,
        help="Whether to save probabilities."
    )
    parser.add_argument(
        "--parallel_jobs",
        required=True,
        help="Number of parallel jobs."
    )

    args = parser.parse_args()

    
    try:
        try:
            slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # Should be 8
        except ValueError:
            slurm_cpus_per_task = 1 # Fallback

        num_concurrent_main_py = args.parallel_jobs # Should be 4

        cpus_for_this_ray_instance = int(max(1, float(slurm_cpus_per_task) // int(num_concurrent_main_py))) # Will be 8 // 4 = 2
        gpus_for_this_ray_instance = float(1 / int(num_concurrent_main_py))  # Assuming 1 GPU per main.py, so 1/4 = 0.25

        log.info("Initializing Ray...")
        logging.getLogger("ray").handlers.clear()
        ray_stdout_handler = logging.StreamHandler(sys.stdout)
        ray_stdout_handler.setLevel(logging.INFO)
        ray_stdout_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger("ray").addHandler(ray_stdout_handler)
        logging.getLogger("ray").setLevel(logging.INFO)
        
        log.info(f"SLURM_CPUS_PER_TASK: {slurm_cpus_per_task}, Concurrent main.py (parallel_jobs arg): {num_concurrent_main_py}, CPUs for this Ray: {cpus_for_this_ray_instance}")

    except Exception as e:
        log.error(f"Failed to get SLURM_CPUS_PER_TASK or parallel_jobs: {e}", exc_info=True)
        sys.exit(1)

    try:
        # Check RAY_TMPDIR environment variable
        ray_tmpdir = os.environ.get("RAY_TMPDIR", "Not set")
        ray_socket_dir = os.environ.get("RAY_SOCKET_DIR", os.path.join(ray_tmpdir, "s") if ray_tmpdir != "Not set" else None)
        
        log.info(f"RAY_TMPDIR from environment: {ray_tmpdir}")
        log.info(f"RAY_SOCKET_DIR from environment: {ray_socket_dir}")
        
        if ray_tmpdir == "Not set" or ray_socket_dir is None:
            log.error("RAY_TMPDIR or RAY_SOCKET_DIR not properly set")
            raise ValueError("Environment variables for Ray not set properly")
            
        # Ensure the socket directory exists (should be created in bash script)
        if not os.path.exists(ray_socket_dir):
            os.makedirs(ray_socket_dir, exist_ok=True)
            log.info(f"Created Ray socket directory: {ray_socket_dir}")
        
        # Test file creation in socket directory
        test_file = os.path.join(ray_socket_dir, "test_socket")
        with open(test_file, "w") as f:
            f.write("Socket test")
        os.remove(test_file)
        log.info(f"Socket directory test successful: {ray_socket_dir}")
        
        # Initialize Ray with explicit socket paths
        ray.init(
            logging_level=logging.INFO,
            include_dashboard=False,
            num_cpus=cpus_for_this_ray_instance,
            num_gpus=1,
            _plasma_directory=ray_socket_dir,  # Use the very short socket directory
            _temp_dir=ray_tmpdir,  # Use the main temporary directory
            enable_resource_isolation=True,
            system_reserved_cpu = cpus_for_this_ray_instance,
            _enable_object_reconstruction=True,
        )

        log.info("Ray initialized successfully")
        
    except Exception as e:
        log.error(f"Failed to initialize Ray: {e}", exc_info=True)
        sys.exit(1)


    # --- Build the experiment_config dictionary ---
    experiment_config = {}
    if hasattr(args,"save_parameters"):
        val = getattr(args, "save_parameters")
        # Handle the boolean value for save_parameters
        if isinstance(val, str):
            if val.lower() == 'true':
                save_parameters = True
            elif val.lower() == 'false':
                save_parameters = False
            else:
                log.error(f"Invalid value for save_parameters: {val}")
                sys.exit(1)
        else:
            save_parameters = bool(val)
    
    if hasattr(args,"save_probs"):
        val = getattr(args, "save_probs")
        # Handle the boolean value for save_probs
        if isinstance(val, str):
            if val.lower() == 'true':
                save_probs = True
            elif val.lower() == 'false':
                save_probs = False
            else:
                log.error(f"Invalid value for save_probs: {val}")
                sys.exit(1)
        else:
            save_probs = bool(val)

    if hasattr(args,"parallel_jobs"):
        val = getattr(args, "parallel_jobs")
        # Handle the integer value for parallel_jobs
        if isinstance(val, str):
            try:
                parallel_jobs = int(val)
            except ValueError:
                log.error(f"Invalid integer value for parallel_jobs: {val}")
                sys.exit(1)
        else:
            parallel_jobs = int(val)

    # Use config_structure_list to ensure all expected keys are present
    for key in config_structure_list:
        if hasattr(args, key):
            val = getattr(args, key)
            if key == 'batch_size_config':
                batch_str = str(val)  # Ensure it's a string first
                if args.batch_size_type == "fixed":
                    try:
                        typed_batch_size = int(batch_str)
                    except ValueError:
                        log.error(f"Invalid integer value for batch size: {batch_str}")
                        sys.exit(1)
                elif args.batch_size_type == "percentage":
                    try:
                        typed_batch_size = float(batch_str)
                    except ValueError:
                        log.error(f"Invalid float value for batch size percentage: {batch_str}")
                        sys.exit(1)
                else :
                    log.error(f"Unsupported batch size type: {args.batch_size_type}")
                    sys.exit(1)
                    
                experiment_config[key] = typed_batch_size  # Assign the typed value

            # Special handling for partitioner_parameter based on partitioner_name
            if key == "partitioner_parameter":
                param_str = str(val) # Ensure it's a string first
                partitioner_name = args.partitioner_name # Get the partitioner name

                if partitioner_name == "dirichlet":
                    try:
                        typed_partitioner_parameter = float(param_str)
                    except ValueError:
                        log.error(f"Invalid float value for dirichlet alpha: {param_str}")
                        sys.exit(1) # Exit if conversion fails
                elif partitioner_name == "pathological":
                    try:
                        typed_partitioner_parameter = int(param_str)
                    except ValueError:
                        log.error(f"Invalid integer value for pathological parameter: {param_str}")
                        sys.exit(1) # Exit if conversion fails
                elif param_str.lower() == 'none':
                    typed_partitioner_parameter = None
                else:
                    # Keep as string or handle other partitioners if necessary
                    typed_partitioner_parameter = param_str
                    log.warning(f"Partitioner parameter '{param_str}' kept as string for partitioner '{partitioner_name}'.")

                experiment_config[key] = typed_partitioner_parameter # Assign the typed value

            # Handle string "None" passed from bash for other optional args
            elif isinstance(val, str) and val.lower() == 'none':
                experiment_config[key] = None
            else:
                experiment_config[key] = val

    # Ensure partitioner_parameter was processed if expected
    if "partitioner_parameter" not in experiment_config:
         log.error("partitioner_parameter was not processed correctly.")
         sys.exit(1)

    try:
        experiment = FLWRExperiment(early_stopping_rounds=5,
                                    ROOT_DIR=ROOT_DIR,
                                    experiment_config=experiment_config,
                                    save_parameters=save_parameters,
                                    save_probs=save_probs,
                                    parallel_jobs=parallel_jobs,)

        experiment.run_experiment()
                
        log.info("Experiment finished successfully.")

        log.info("Shutting down Ray...")
        ray.shutdown()
        log.info("Ray shut down.")

    except Exception as e:
        log.error(f"Experiment failed: {e}", exc_info=True) # Log traceback

        # Attempt to delete any partial results
        try:
            from fl_flwr_Results_Manager import ResultsManager
            results_manager = ResultsManager(config=experiment_config)
            results_manager.delete_experiment_node(experiment_config)
            log.info("Cleaned up partial experiment results")
        except Exception as cleanup_err:
            log.error(f"Failed to clean up results: {cleanup_err}")
        
        # Force Ray shutdown
        try:
            ray.shutdown()
        except:
            pass
        
        # Exit with error code
        sys.exit(1)




if __name__ == "__main__": # Important: set_start_method should be in the main block
    try:
        mp.set_start_method('spawn', force=True) # Add this line
    except RuntimeError:
        pass # If already set, it might raise a RuntimeError
    main() # Your existing main function call