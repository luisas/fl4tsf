
process FEDERATED_TRAINING {

    label "process_medium"
    tag "$meta.id - ${meta.aggregation}"
    
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_matplotlib_pruned:c1a4d380c9f71c94' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"


    input:
    tuple val(meta), path(data)
    path(bin)
    path(config)

    output:
    tuple val(meta), path("federated_outputs/*.json"), emit: metrics
    tuple val(meta), path("federated_outputs/*.pth") , emit: model
    path("federated_outputs/meta.csv")               , emit: meta_csv
    //path("weights*.pt")                              , emit: weights, optional: true

    when:
    task.ext.when == null || task.ext.when

    script:
    def keys = meta.keySet().join(",")
    def values = meta.values().join(",")
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    export PYTHONUNBUFFERED=1
    export RAY_LOG_TO_STDERR=0
    export RAY_LOG_TO_FILE=1
    export RAY_LOG_DIR="./ray_logs"


    # If NOT running on SLURM (likely local)
    # TODO: this is a quick fix and only supports slurm executor but should be made cleaner
    if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
        ulimit -u 10000
    fi

    # Setup Ray environment
    export RAY_TMPDIR="/tmp/ray_tmp_luisa/\$RANDOM"
    mkdir -p "\$RAY_TMPDIR"
    export RAY_SOCKET_DIR="/tmp/ray_tmp_luisa/\$RANDOM/s"
    export RAY_object_store_memory=10737418240
    mkdir -p "\$RAY_SOCKET_DIR"

    python main.py --ncpus ${task.cpus} --ngpus ${task.accelerator.request} --raydir \$RAY_TMPDIR --ray_socket_dir \$RAY_SOCKET_DIR --nclients ${meta.clients}
    
    # Store a file with all the meta information
    echo "$keys" > meta.csv
    echo "$values" >> meta.csv
    mv meta.csv federated_outputs/meta.csv
    """
}