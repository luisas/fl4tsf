
process FEDERATED_TRAINING {
    tag "$meta.id"
    label 'process_low'
    

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    input:
    tuple val(meta), path(data)
    path(bin)

    output:
    tuple val(meta), path("federated_outputs/*.json"), emit: metrics
    tuple val(meta), path("federated_outputs/*.pth") , emit: model

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    flwr run .
    """
}