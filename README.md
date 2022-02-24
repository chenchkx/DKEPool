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
open run_mutag.sh file

DKE_TYPE="${5-0}"
sh run_mutag.sh
```

For robust DKEPool module tested on MUTAG dataset, run

```
open run_mutag.sh file

DKE_TYPE="${5-1}"
sh run_mutag.sh
```
