flwr run . --run-config "num-server-rounds=5 \
                fraction-fit=1 \
                fraction-evaluate=1 \
                local-epochs=2 \
                batch-size=50 \ 
                learning-rate=0.01 " 
