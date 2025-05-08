
include { CENTRALIZED_TRAINING } from '../modules/centralized_training/main.nf'
include { FEDERATED_TRAINING   } from '../modules/federated_training/main.nf'


workflow FEDERATED_LEARNING_SIMULATION{

    take: 
    centralized_data_and_params_ch
    federated_data_and_params_ch
    bin

    main:
    //CENTRALIZED_TRAINING(centralized_data_and_params_ch)

    FEDERATED_TRAINING(federated_data_and_params_ch, bin)

}