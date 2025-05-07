

process CENTRALIZED_TRAINING {
    tag "$meta.id"
    label 'process_low'
    

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("*.csv"), emit: metrics
    tuple val(meta), path("*.pth"), emit: model

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    centralized_train.py
    """
}