#python centralized_train.py --dataset physionet_filtered --sample_tp 0.5 --batch_size 50 --epochs 10 --lr 0.01  --dataset_name "../data/physionet_filtered/client_0" --output_dir "../results_test/"

python centralized_train.py --dataset ecg_subset --sample_tp 0.5 --batch_size 32 --epochs 30 --lr 0.001  --dataset_name "../data/ecg_subset/client_0" --output_dir "../results_centralized_ecgsubset/"