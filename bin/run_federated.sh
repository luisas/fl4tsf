
# create unique nr 
export PYTHONUNBUFFERED=1


# # Setup Ray environment
# export RAY_TMPDIR="/tmp/ray_tmp_luisa/$RANDOM/"
# export RAY_object_store_memory=5737418240

# # Create the directory first
# mkdir -p "${RAY_TMPDIR}"
# mkdir -p "${RAY_TMPDIR}/s"

# # Export the variables for the Python script
# export RAY_SOCKET_DIR="${RAY_TMPDIR}/s"


# Setup Ray environment
export RAY_TMPDIR="/tmp/ray_tmp_luisa/$RANDOM"
mkdir -p "$RAY_TMPDIR"
export RAY_SOCKET_DIR="/tmp/ray_tmp_luisa/$RANDOM/s"
export RAY_object_store_memory=10737418240
mkdir -p "$RAY_SOCKET_DIR"

python main.py --ncpus 2 --ngpus 0 --raydir $RAY_TMPDIR --ray_socket_dir $RAY_SOCKET_DIR --nclients 2


