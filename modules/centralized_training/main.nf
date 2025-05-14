

process CENTRALIZED_TRAINING {
    tag "$meta.id"
    label 'process_low'
    

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    input:
    tuple val(meta), path(data)
    path(config)

    output:
    tuple val(meta), path("*.csv"), emit: metrics
    tuple val(meta), path("*.pth"), emit: model
    path(meta.csv)                , emit: meta_csv

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def keys = meta.keySet().join(",")
    def values = meta.values().join(",")
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    centralized_train.py --epochs ${meta.epochs} \\
                         --lr ${meta.lr} \\
                         --batch_size ${meta.batch_size} \\
                         --dataset ${meta.id} \\
                         --sample_tp ${meta.sample_tp}

    # data in the current directory
    echo "$keys" > meta.csv
    echo "$values" >> meta.csv
    """
}