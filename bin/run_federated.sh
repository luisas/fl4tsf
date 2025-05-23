
# create unique nr 
export PYTHONUNBUFFERED=1


# Setup Ray environment
export RAY_TMPDIR="/tmp/ray_tmp_luisa/$RANDOM/"
export RAY_object_store_memory=5737418240

# Create the directory first
mkdir -p "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}/s"

# Export the variables for the Python script
export RAY_SOCKET_DIR="${RAY_TMPDIR}/s"

# Run the Python script
python main.py --ncpus 2

