
include { CENTRALIZED_TRAINING } from '../modules/centralized_training/main.nf'


workflow FEDERATED_LEARNING_SIMULATION{

    take: 
    input_ch

    main:
    CENTRALIZED_TRAINING(input_ch)


}