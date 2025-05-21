
include { CENTRALIZED_TRAINING                    } from '../modules/centralized_training/main.nf'
include { FEDERATED_TRAINING                      } from '../modules/federated_training/main.nf'
include { PREPARE_MODEL_CONFIG as PREP_CONFIG_FED } from '../modules/prepare_model_config/main.nf'
include { PREPARE_MODEL_CONFIG as PREP_CONFIG_CEN } from '../modules/prepare_model_config/main.nf'


workflow FEDERATED_LEARNING_SIMULATION{

    take: 
    centralized_data_and_params_ch
    federated_data_and_params_ch
    bin

    main:

    if(!params.skip_centralized){
        cen_meta = centralized_data_and_params_ch.map{ meta, data -> meta}
        PREP_CONFIG_CEN(cen_meta)
        cen_model_config = PREP_CONFIG_CEN.out.config
        CENTRALIZED_TRAINING(centralized_data_and_params_ch, cen_model_config )
    }


    if (!params.skip_federated){
        fed_meta = federated_data_and_params_ch.map{ meta, data -> meta}
        PREP_CONFIG_FED(fed_meta)
        fed_model_config = PREP_CONFIG_FED.out.config
        FEDERATED_TRAINING(federated_data_and_params_ch, bin, fed_model_config)
    }


}