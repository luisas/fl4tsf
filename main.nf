

include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'



// Create a dummy meta channel


workflow {


    main:

    // Load dataset 
    Channel
        .fromPath("${projectDir}/data/periodic/*")
        .collect()
        .map{ dir -> 
                [[id: "test"], dir]
        }  
        .set { training_data_ch }

    training_data_ch.view()

    FEDERATED_LEARNING_SIMULATION(training_data_ch)

}