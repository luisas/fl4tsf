

include {  FEDERATED_LEARNING_SIMULATION } from './workflows/federated_learning_simulation.nf'



// Create a dummy meta channel


workflow {


    main:

    // Load dataset 
    Channel
        .fromPath("${projectDir}/data/${params.dataset}/*")
        .collect()
        .map{ dir -> 
                [[id: "${params.dataset}", epochs: "${params.c_epochs}"], dir]
        }  
        .set { training_data_ch }

    // Load bin
    Channel
        .fromPath("${projectDir}/bin/*")
        .collect()
        .set { bin_ch }

    FEDERATED_LEARNING_SIMULATION(training_data_ch, bin_ch)

}