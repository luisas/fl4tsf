# FL4TSF


main pipeline:
```
main.nf
```

## analysis notebook 

#### 1 - generate simulated dataset

```
cd bin
python create_periodic.py
```

```
analysis/01_dataset_visualization.ipynb
```

#### 2 - centralized training

- explore hyperparameters for centralized learning
- sanity check that the set up model can learn 

```
-profile centralized_best
-profile centralized_hyperparam
```

```
analysis/02_centralized_training.ipynb
```

#### 3 - federated training IID

- sanity check: federated training with FedAvg in IID setting. 

```
analysis/03_federated_training.ipynb
```