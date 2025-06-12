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

#### 2 - centralized training

- explore hyperparameters for centralized learning
- sanity check that the set up model can learn 

```
-profile centralized_best
-profile centralized_hyperparam
```