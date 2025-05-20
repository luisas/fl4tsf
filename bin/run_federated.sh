
# create unique nr 
export PYTHONUNBUFFERED=1

# Setup Ray environment
export RAY_TMPDIR="/tmp/ray_tmp_luisa/"
export RAY_object_store_memory=10737418240
export RAY_SOCKET_DIR="/tmp/ray_tmp_luisa/s"

python main.py
