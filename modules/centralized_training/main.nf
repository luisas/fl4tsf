

process CENTRALIZED_TRAINING {
    tag "$meta.id"
    label 'process_low'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("outputs/*"), emit: metrics

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    mkdir -p \$MPLCONFIGDIR
    echo "running in the container"
    mkdir -p outputs

    centralized_train.py
    """
}