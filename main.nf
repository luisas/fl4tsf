

include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'



// Create a dummy meta channel


workflow {


    main:

    // Take a param epochs and split it with a comma
    // These are the only parameters that we can run multiple values of
    // The rest of the parameters are set to a single value in the config file
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


    // Replicate is a special case, we want to run it multiple times and given by number, create list of numbers with max params.replicate
    def replicate       = (1..Integer.parseInt("${params.replicate}")).collect { it.toString() }



    // General parameters 
    Channel
        .of(lr)
        .combine(Channel.from(batch_size))
        .combine(Channel.from(clipping))
        .combine(Channel.from(lrdecay))
        .map{
            lr_val, bs, cl, lrd->
                [
                    id          : "${params.dataset}",
                    dataset_name: "${params.dataset}",
                    data_folder : ".", 
                    lr          : lr_val,
                    lrdecay     : lrd,
                    batch_size  : bs,
                    sample_tp   : "${params.sample_tp}",
                    cut_tp      : "${params.cut_tp}",
                    extrap      : "${params.extrap}", 
                    gradientclipping: cl,
                ]
        }.set { meta_general }


    // Create cross product of epochs and lr
    Channel
        .from(epochs)
        .map { epoch ->
            [
                epochs      : epoch
            ]
        }
        .set { meta_centralized }
    

    // Model parameters
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


    // Federated meta channel
    Channel
        .of(serverrounds)
        .combine(Channel.from(aggregation))
        .combine(Channel.from(alpha))
        .combine(Channel.from(clients))
        .combine(Channel.from(replicate))
        .combine(Channel.from(localepochs))
        .map({
                sr, ag, alpha_val, cl, rep, le->
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
                use_wandb: "${params.use_wandb}",]
            })
        .set { meta_federated }

    meta_general
        .combine(meta_model_params)
        .combine(meta_centralized)
        .map{ meta, meta2, meta3 -> 
                [meta + meta2 + meta3]
        }
        .set { meta_centralized }
    
    meta_general
        .combine(meta_model_params)
        .combine(meta_federated)
        .map{ meta, meta2, meta3 -> 
                [meta + meta2 + meta3]
        }
        .set { meta_federated }
    

    // Load dataset 
    Channel
        .fromPath("${projectDir}/data/${params.dataset}/*")
        .collect()
        .map{ dir -> 
                [[id: "${params.dataset}"], dir]
        }  
        .set { training_data_ch }

    // Load bin
    Channel
        .fromPath("${projectDir}/bin/*", type: 'any')
        .filter { it.name != 'model.config' && it.name != '__pycache__' && it.name != 'federated_outputs' }
        .collect()
        .set { bin_ch }


    // Prepare centralized training data
    training_data_ch.combine(meta_centralized)
        .map{ meta, data, meta2 -> 
                [meta + meta2, data]
        }
        .set { centralized_data_and_params_ch }


    // Prepare federated training data
    training_data_ch.combine(meta_federated)
        .map{ meta, data, meta2 -> 
                [meta + meta2, data]
        }
        .set { federated_data_and_params_ch }

    FEDERATED_LEARNING_SIMULATION(centralized_data_and_params_ch, federated_data_and_params_ch, bin_ch)

}