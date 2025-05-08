
include { CENTRALIZED_TRAINING } from '../modules/centralized_training/main.nf'
include { FEDERATED_TRAINING   } from '../modules/federated_training/main.nf'


workflow FEDERATED_LEARNING_SIMULATION{

    take: 
    input_ch
    bin

    main:
    CENTRALIZED_TRAINING(input_ch)

    FEDERATED_TRAINING(input_ch, bin)

}