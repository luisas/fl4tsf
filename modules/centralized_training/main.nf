

process CENTRALIZED_TRAINING {
    tag "$meta.id"
    label 'process_low'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-27978155697a3671f3ef9aead4b5c823a02cc0b7:548df772fe13c0232a7eab1bc1deb98b495a05ab-0' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:9b2f5640408457e9' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("placeholder.txt"), emit: metrics

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    centralized_train.py
    touch placeholder.txt
    """
}