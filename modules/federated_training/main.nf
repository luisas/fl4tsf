
process FEDERATED_TRAINING {
    tag "$meta.id - ${meta.aggregation}"
    label 'process_low'
    

    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_matplotlib_pruned:c1a4d380c9f71c94' :
    //     'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"


    input:
    tuple val(meta), path(data)
    path(bin)
    path(config)

    output:
    tuple val(meta), path("federated_outputs/*.json"), emit: metrics
    tuple val(meta), path("federated_outputs/*.pth") , emit: model

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    flwr run . --run-config "num-server-rounds=${meta.serverrounds} \
                    fraction-fit=${meta.fractionfit} \
                    fraction-evaluate=${meta.fractionevaluate} \
                    local-epochs=${meta.localepochs} \
                    batch-size=${meta.batch_size}\ 
                    learning-rate=${meta.lr} " 
    """
}