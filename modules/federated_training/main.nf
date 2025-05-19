
process FEDERATED_TRAINING {
    tag "$meta.id - ${meta.aggregation}"
    
    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_matplotlib_pruned:c1a4d380c9f71c94' :
    //     'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
    //     'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"


    input:
    tuple val(meta), path(data)
    path(bin)
    path(config)

    output:
    tuple val(meta), path("federated_outputs/*.json"), emit: metrics
    tuple val(meta), path("federated_outputs/*.pth") , emit: model
    path("federated_outputs/meta.csv")               , emit: meta_csv

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def keys = meta.keySet().join(",")
    def values = meta.values().join(",")
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    mkdir -p /tmp/ray_tmp
    # create unique nr 
    export PYTHONUNBUFFERED=1

    # Setup Ray environment
    export RAY_TMPDIR="/tmp/ray_tmp_luisa/"
    export RAY_object_store_memory=10737418240

    # Create the directory first
    mkdir -p "\${RAY_TMPDIR}"
    mkdir -p "\${RAY_TMPDIR}/s"

    # Export the variables for the Python script
    export RAY_TMPDIR="\${RAY_TMPDIR}"
    export RAY_SOCKET_DIR="\${RAY_TMPDIR}/s"

    # Run the Python script
    python main.py

    # Store a file with all the meta information
    echo "$keys" > meta.csv
    echo "$values" >> meta.csv
    mv meta.csv federated_outputs/meta.csv
    """
}