

include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'



// Create a dummy meta channel


workflow {


    main:


    // Centralized meta channel
    Channel
        .of([
                [id: "${params.dataset}", 
                epochs: "${params.epochs}", 
                lr: "${params.lr}", 
                batch_size: "${params.batch_size}", 
                sample_tp: "${params.sample_tp}"]
        ])
        .set { meta_centralized }

    // Federated meta channel
    Channel
        .of([
                [id: "${params.dataset}", 
                serverrounds: "${params.serverrounds}", 
                fractionfit: "${params.fractionfit}", 
                fractionevaluate: "${params.fractionevaluate}", 
                localepochs: "${params.localepochs}", 
                batch_size: "${params.batch_size}", 
                lr: "${params.lr}", 
                numsupernodes: "${params.numsupernodes}"]
        ])
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
        .fromPath("${projectDir}/bin/*")
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