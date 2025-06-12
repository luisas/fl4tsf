

process MERGE_CLIENTS_DATA {
    tag "$meta.id"
    label "process_single"
    
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:527707828ce78fbf' :
        'community.wave.seqera.io/library/pip_flwr-datasets_flwr_numpy_pruned:37e97d65f19bcbe8' }"

    input:
    tuple val(meta), path(data)

    output:
    tuple val(meta), path("*.pt")  , emit: data

    when:
    task.ext.when == null || task.ext.when

    script:
    def keys = meta.keySet().join(",")
    def values = meta.values().join(",")
    """
    export MPLCONFIGDIR=\$PWD/.mplconfig
    merge_clients_data.py --prefix ${meta.dataset_name}
    """
}