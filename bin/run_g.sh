#!/bin/bash
#SBATCH --job-name=cifar10_dirichlet_10_0p2_FedAvg_adam_cross_entropy_normbatch_norm_bs0p003_percentage__lr0p0005_fixed__rounds500_le1_fixed_
#SBATCH --output=/gpfs/commons/groups/gursoy_lab/lsantus/out/log.txt
#SBATCH --error=/gpfs/commons/groups/gursoy_lab/lsantus/err/log.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --gres=cpu:1
#SBATCH --time=40:00:00
#SBATCH --exclude=ne1dg6-004


export PYTHONUNBUFFERED=1

# Setup Ray environment
export RAY_TMPDIR="/tmp/ray_temp/r${RANDOM}"
export RAY_object_store_memory=10737418240

# Create the directory first
mkdir -p "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}/s"

# Export the variables for the Python script
export RAY_TMPDIR="${RAY_TMPDIR}"
export RAY_SOCKET_DIR="${RAY_TMPDIR}/s"

# Run the Python script
python main.py --dataset_name cifar10 --partitioner_name dirichlet --partitioner_parameter 0.2 --num_partitions 10 --strategy_name FedAvg --optimizer_name adam --criterion_name cross_entropy --normalization batch_norm --batch_size_type percentage --batch_size_config 0.003 --learning_rate_type fixed --learning_rate_config 0.0005 --num_rounds 500 --local_epochs_type fixed --local_epochs_config 1 --save_parameters False --save_probs True --parallel_jobs 1
