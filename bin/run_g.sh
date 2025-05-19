#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=luisa_test
#SBATCH --output=/gpfs/commons/groups/gursoy_lab/lsantus/out/log.txt
#SBATCH --error=/gpfs/commons/groups/gursoy_lab/lsantus/err/log.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=40:00:00
#SBATCH --exclude=ne1dg6-004


export PYTHONUNBUFFERED=1

# Setup Ray environment
export RAY_TMPDIR="/tmp/ray_tmp_luisa/r${RANDOM}"
export RAY_object_store_memory=10737418240

# Create the directory first
mkdir -p "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}/s"

# Export the variables for the Python script
export RAY_TMPDIR="${RAY_TMPDIR}"
export RAY_SOCKET_DIR="${RAY_TMPDIR}/s"

# Run the Python script
python main.py


