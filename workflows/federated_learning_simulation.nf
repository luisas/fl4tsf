
include { CENTRALIZED_TRAINING as CENTRALIZED_LOCAL_CLIENT_TRAINING  } from '../modules/centralized_training/main.nf'
include { CENTRALIZED_TRAINING as CENTRALIZED_TRAINING               } from '../modules/centralized_training/main.nf'
include { FEDERATED_TRAINING                                         } from '../modules/federated_training/main.nf'
include { MERGE_CLIENTS_DATA                                         } from '../modules/merge_clients_data/main.nf'
include { PREPARE_MODEL_CONFIG as PREP_CONFIG_FED                    } from '../modules/prepare_model_config/main.nf'
include { PREPARE_MODEL_CONFIG as PREP_CONFIG_CEN                    } from '../modules/prepare_model_config/main.nf'
include { PREPARE_MODEL_CONFIG as PREP_CONFIG_CEN_LOCAL              } from '../modules/prepare_model_config/main.nf'


workflow FEDERATED_LEARNING_SIMULATION{

    take: 
    centralized_data_and_params_ch
    centralized_data_and_params_local_ch
    federated_data_and_params_ch
    bin

    main:


    /*
    * Centralized training
    * Currently: merges all clients data into one dataset and trains a model on it.
    */

    if(!params.skip_centralized){

        // Merge all clients data into one dataset
        MERGE_CLIENTS_DATA(centralized_data_and_params_ch)
        MERGE_CLIENTS_DATA.out.data.set{ merged_data }

        // Prepare model config for centralized training
        cen_meta = merged_data.map{ meta, _data -> meta}
        PREP_CONFIG_CEN(cen_meta)
        cen_model_config = PREP_CONFIG_CEN.out.config

        // combine centralized_data_and_params_ch and cen_model_config
        merged_data.combine(cen_model_config, by: 0).multiMap{ meta, data, config -> 
            dataset: [meta, data]
            conf: config
        }.set { cen_training_input }

        // Launch centralized training
        CENTRALIZED_TRAINING(cen_training_input.dataset, cen_training_input.conf)
    }


    /*
    * Centralized training for local clients
    * Each element in the channel corresponds to one client so it trains a model on each client's data separately.
    */

    if(!params.skip_local){

        // Prepare config
        cen_meta = centralized_data_and_params_local_ch.map{ meta, _data -> meta}
        PREP_CONFIG_CEN_LOCAL(cen_meta)
        cen_model_config = PREP_CONFIG_CEN_LOCAL.out.config

        // combine centralized_data_and_params_ch and cen_model_config
        centralized_data_and_params_local_ch.combine(cen_model_config, by: 0).multiMap{ meta, data, config -> 
            dataset: [meta, data]
            conf: config
        }.set { cen_training_input }

        // Launch centralized training for local clients
        CENTRALIZED_LOCAL_CLIENT_TRAINING(cen_training_input.dataset, cen_training_input.conf)
    }


    /*
    * Federated training with flower     
    */
    if (!params.skip_federated){

        // Prepare config
        fed_meta = federated_data_and_params_ch.map{ meta, _data -> meta}
        PREP_CONFIG_FED(fed_meta)
        fed_model_config = PREP_CONFIG_FED.out.config

        // combine federated_data_and_params_ch and fed_model_config\
        federated_data_and_params_ch.combine(fed_model_config, by: 0).multiMap{ meta, data, config -> 
            dataset: [meta, data]
            conf: config
        }.set { fed_training_input }

        // Launch federated training
        FEDERATED_TRAINING(fed_training_input.dataset, bin, fed_training_input.conf)
    }


}