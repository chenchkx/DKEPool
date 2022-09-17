# Distribution Knowledge Embedding for Graph Pooling

This is the code for our paper "Distribution Knowledge Embedding for Graph Pooling". It is based on the code from [SOPool](https://github.com/divelab/sopool). Many thanks!

Created by [Kaixuan Chen](chenkx@zju.edu.cn) (chenkx@zju.edu.cn, chenkx.jsh@aliyun.com)

## Download & Citation

If you find our code useful for your research, please kindly cite our paper.

```
@article{chen2021distribution,
  title={Distribution Knowledge Embedding for Graph Pooling},
  author={Chen, Kaixuan and Song, Jie and Liu, Shunyu and Yu, Na and Feng, Zunlei and Han, Gengshi and Song, Mingli},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022}
}
```

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
set DKE_TYPE="${5-0}"

sh run_mutag.sh
```

For robust DKEPool module tested on MUTAG dataset, run

```
open run_mutag.sh file
set DKE_TYPE="${5-1}"

sh run_mutag.sh
```
