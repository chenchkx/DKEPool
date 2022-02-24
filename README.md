# Distribution Knowledge Embedding for Graph Pooling

## System requirement

#### Programming language
```
Python 3.6
```
#### Python Packages
```
PyTorch > 1.0.0, tqdm, networkx, numpy
```

## Run the code

We provide scripts to run the experiments. 

For DKEPool module tested on MUTAG dataset, run

```
chmod +x run_bio.sh
./run_bio.sh [DATASET] [GPU_ID] [BATCH_SIZE] [HIDDEN_DIM]
```

For robust DKEPool module tested on MUTAG dataset, run
```
chmod +x run_social.sh
./run_social.sh [DATASET] [GPU_ID] [BATCH_SIZE] [HIDDEN_DIM]
```
