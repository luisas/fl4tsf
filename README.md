
# Federated Learning for Time Series Forecasting with Latent Neural ODEs  


## Nextflow <img src="https://avatars.githubusercontent.com/u/6698688?s=200&v=4" alt="Flower" style="width:30px; height:30px;"/> for Flower <img src="https://flower.ai/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fflwr-head.4d68867a.png&w=384&q=75" alt="Flower Website" style="width:30px; height:30px;" />



## Overview  
**FL4TSF** is a dual-purpose repository that brings together:  

1. **A Nextflow + Flower framework** for the **systematic exploration of federated learning (FL)** in reproducible, configurable workflows.
  - Test several federated learning configuration settings and systematically explore the effects on final local, federated, and centralized performance
2. **A testbed for Latent Neural ODEs (Neural Ordinary Differential Equations)** applied to time-series forecasting for healthcare
 - For example, we use it to explore novel custom aggregation functions specific to neural ODEs



## How to use it: 

main pipeline:
```
main.nf
```

```
nextflow run main.nf -profile <YOUR_PROFILE>,singularity
```
