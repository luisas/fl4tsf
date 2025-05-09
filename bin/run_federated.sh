flwr run . --run-config "num-server-rounds= 2 \
                fraction-fit=1 \
                fraction-evaluate=1 \
                local-epochs=1 \
                batch-size=50 \ 
                learning-rate=0.01 " 

mv federated_outputs ../results_local