# coding=utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
# from models.graphcnn import GraphCNN
# from sopool_attn.graphcnn import GraphCNN
from dkepool.graphcnn import GraphCNN


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)



    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
parser = argparse.ArgumentParser(
    description='PyTorch graph convolutional neural net for whole-graph classification')
parser.add_argument('--dataset', type=str, default="IMDBBINARY",
                    help='name of dataset (default: REDDITBINARY)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--iters_per_epoch', type=int, default=50,
                    help='number of iterations per each epoch (default: 50)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 320050)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--fold_idx', type=int, default=0,
                    help='the index of fold in 10-fold validation. Should be less then 10.')
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of layers INCLUDING the input one (default: 5)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
parser.add_argument('--hidden_dim', type=int, default=32,
                    help='number of hidden units (default: 32)')
parser.add_argument('--final_dropout', type=float, default=0.5,
                    help='final layer dropout (default: 0.5)')
parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                    help='Pooling for over nodes in a graph: sum or average')
parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                    help='Pooling for over neighboring nodes: sum, average or max')
parser.add_argument('--learn_eps', action="store_true",
                    help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
parser.add_argument('--degree_as_tag', action="store_true",
                    help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
parser.add_argument('--fac_dim', type=int, default=1,
                    help='dimensionality of the factorized matrices')
parser.add_argument('--rep_dim', type=int, default=1000,
                    help='dimensionality of the representations')
parser.add_argument('--filename', type=str, default="10flod.txt",
                    help='output file')
args = parser.parse_args()

# set up seeds and gpu device
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(10) #设置CPU占用核数
# device = torch.device("cpu")
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

graphs, num_classes = load_data(args.dataset, args.degree_as_tag)


sta_num = 1000
sta_array = np.zeros([sta_num, 5000])
for i in range(sta_num):
    g = graphs[i]
    neighbor = g.neighbors
    for j in range(len(neighbor)):
        node_edges = neighbor[j]
        num_edges = len(node_edges)
        sta_array[i,num_edges] += 1


a = np.mean(sta_array,axis=0)
##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)




