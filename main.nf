

include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'



// Create a dummy meta channel


workflow {


    main:

    // Load dataset 
    Channel
        .fromPath("${projectDir}/data/${params.dataset}/*")
        .collect()
        .map{ dir -> 
                [[id: "${params.dataset}"], dir]
        }  
        .set { training_data_ch }

    training_data_ch.view()

    FEDERATED_LEARNING_SIMULATION(training_data_ch)

}