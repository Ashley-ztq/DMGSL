import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

import pandas as pd


warnings.simplefilter("ignore")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip())) # s.strip(rm) 删除s字符串中开头、结尾处，位于 rm删除序列的字符 当rm为空时，默认删除空白符(包括'\n', '\r', '\t', ' ')
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_citation_network():

    feature = pd.read_csv("./data/normalized_feature_ul.csv", dtype='float16', index_col=0)
    feature = np.nan_to_num(feature)
    features = feature.T
    adj = pd.read_csv("./data/adj_ul_edgetype.csv", header=0, index_col=0)
    adj = np.array(adj, dtype='float32')
    labels = pd.read_csv('./data/node_label.csv',header = None,index_col = None)
    labels = np.array(labels)

    idx_test = pd.read_csv('./data/test_index.csv', header=None)
    idx_train = pd.read_csv('./data/train_index.csv', header=None)
    idx_val = pd.read_csv('./data/val_index.csv', header=None)

    idx_test = np.array(idx_test)
    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


def load_data(arg):
    return load_citation_network()
