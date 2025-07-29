include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'

workflow {

    main:

    // Split the input values of the parameters that support multiple values
    def datasets        = "${params.dataset}".split(",")
    def epochs          = "${params.epochs}".split(",")
    def lr              = "${params.lr}".split(",")
    def batch_size      = "${params.batch_size}".split(",")
    def serverrounds    = "${params.serverrounds}".split(",")
    def aggregation     = "${params.aggregation}".split(",")
    def alpha           = "${params.alpha}".split(",")
    def clients         = "${params.clients}".split(",")
    def clipping        = "${params.gradient_clipping}".split(",")
    def lrdecay         = "${params.lrdecay}".split(",")
    def localepochs     = "${params.localepochs}".split(",")
    def decay_onset     = "${params.decay_onset}".split(",")
    def use_jit         = "${params.use_jit}".split(",")


    // Replicate is a special case, we want to run it multiple times and given by number, create list of numbers with max params.replicate
    def replicate       = (1..Integer.parseInt("${params.replicate}")).collect { it.toString() }

    // General parameters (common to all cetralized, federated, and local training)
    Channel
        .of(lr)
        .combine(Channel.from(batch_size))
        .combine(Channel.from(clipping))
        .combine(Channel.from(lrdecay))
        .map{
            lr_val, bs, cl, lrd->
                [
                    data_folder : ".", 
                    lr          : lr_val,
                    lrdecay     : lrd,
                    batch_size  : bs,
                    sample_tp   : "${params.sample_tp}",
                    cut_tp      : "${params.cut_tp}",
                    extrap      : "${params.extrap}", 
                    gradientclipping: cl,
                    storeweights: "${params.storeweights}"
                ]
        }.set { meta_general }

    // Model parameters (common to all cetralized, federated, and local training)
    Channel
        .of([
                [obsrv_std: "${params.obsrv_std}",
                poisson: "${params.poisson}",
                rec_layers: "${params.rec_layers}",
                gen_layers: "${params.gen_layers}",
                units: "${params.units}",
                gru_units: "${params.gru_units}",
                latents: "${params.latents}",
                rec_dims: "${params.rec_dims}",
                z0_encoder: "${params.z0_encoder}",
                train_classif_w_reconstr: "${params.train_classif_w_reconstr}",
                classif: "${params.classif}",
                linear_classif: "${params.linear_classif}",
                classif_per_tp: "${params.classif_per_tp}",
                n_labels: "${params.n_labels}",
                input_dim: "${params.input_dim}"]
        ])
        .set { meta_model_params }


    // Parameters for centralized training
    // Currently this is also the one used for local training, this may have to be changed in the future
    Channel
        .from(epochs)
        .map { epoch ->
            [
                epochs      : epoch
            ]
        }
        .set { meta_centralized }


    // Parameters for federated training
    Channel
        .of(serverrounds)
        .combine(Channel.from(aggregation))
        .combine(Channel.from(alpha))
        .combine(Channel.from(clients))
        .combine(Channel.from(replicate))
        .combine(Channel.from(localepochs))
        .combine(Channel.from(decay_onset))
        .map({
                sr, ag, alpha_val, cl, rep, le, deco ->
                [
                obsrv_std: "${params.obsrv_std}", 
                serverrounds: sr, 
                fractionfit: "${params.fractionfit}", 
                fractionevaluate: "${params.fractionevaluate}", 
                localepochs: le, 
                numsupernodes: "${params.numsupernodes}", 
                aggregation: ag, 
                alpha: alpha_val, 
                clients: cl, 
                replicate: rep, 
                decay_onset: deco,
                use_wandb: "${params.use_wandb}",]
            })
        .set { meta_federated }

    // Prepare meta for centralized training
    meta_general
        .combine(meta_model_params)
        .combine(meta_centralized)
        .map{ meta, meta2, meta3 -> 
                [meta + meta2 + meta3]
        }
        .set { meta_centralized }
    
    // Prepare meta for federated training
    meta_general
        .combine(meta_model_params)
        .combine(meta_federated)
        .map{ meta, meta2, meta3 -> 
                [meta + meta2 + meta3]
        }
        .set { meta_federated }
    

    // Collect training data according to the dataset parameter
    // Collects all client files for a given dataset name
    Channel
        .from(datasets)
        .map { ds ->
            def files = file("${params.data_folder}/${ds}*/*")
            def grouped = files.groupBy { f -> f.getParent().getName() }
            return grouped.collect { id, fileList ->
                [ [id: id, dataset_name: id], fileList ]
            }
        }
        .flatMap()
        .set { training_data_ch }


    // Separate the training data by client for the local training
    // There each module will have access to the files of only one client
    training_data_ch
        .map { meta, files ->
            // Group files by client
            def client_files = [:]
            files.each { file ->
                def filename = file.name
                def client_match = filename =~ /(client_\d+)/
                if (client_match) {
                    def client_id = client_match[0][1]
                    if (!client_files.containsKey(client_id)) {
                        client_files[client_id] = []
                    }
                    client_files[client_id] << file
                }
            }
            return [meta, client_files]
        }
        .flatMap { meta, client_files ->
            // Create separate entries for each client
            client_files.collect { client_id, files ->
                def new_meta = meta + [dataset_name: client_id]
                [new_meta, files]
            }
        }
        .set { training_data_clients_ch }

    // Load bin
    Channel
        .fromPath("${projectDir}/bin/*", type: 'any')
        .filter { it.name != 'model.config' && it.name != '__pycache__' && it.name != 'federated_outputs' }
        .collect()
        .set { bin_ch }


    // Prepare centralized training data and parameters instructions
    training_data_ch.combine(meta_centralized)
        .map{ meta, data, meta2 -> 
                [meta + meta2, data]
        }
        .set { centralized_data_and_params_ch }

    // Prepare centralized training data and parameters instructions for local training
    training_data_clients_ch.combine(meta_centralized)
        .map{ meta, data, meta2 -> 
                [meta + meta2, data]
        }
        .set { centralized_data_and_params_local_ch }

    // Prepare federated training data and parameters instructions
    training_data_ch.combine(meta_federated)
        .map{ meta, data, meta2 -> 
                [meta + meta2, data]
        }
        .set { federated_data_and_params_ch }

    // Launch simulation
    FEDERATED_LEARNING_SIMULATION(centralized_data_and_params_ch, centralized_data_and_params_local_ch, federated_data_and_params_ch, bin_ch)

}