import copy
import math

import torch

from graph_generator import *
from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU
from edgeattention import Edgeattention

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse
        self.softmax = nn.Softmax()

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):

        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)   # 激活函数
            x = F.dropout(x, p=self.dropout, training=self.training)
            #x = self.softmax(x) ##############
        x = self.layers[-1](x, Adj)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()   # 它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))  # Sequential像是一个容器包装各层

    def forward(self, x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            ######### 自己添加
            #bn = torch.nn.BatchNorm1d(x.shape[1]).cuda()
            #x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()

        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)
        ## 下面为自己添加的代码
        #self.attention = Edgeattention(256, 128)
        ##
    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        ## 下面为自己增加的代码
        #z1, embedding1 = self.encoder(x, Adj_1, branch)
        #z2, embedding2 = self.encoder(x, Adj_2, branch)
        #z3, embedding3 = self.encoder(x, Adj_3, branch)
        #inputs = torch.stack([embedding1, embedding2, embedding3])
        #z = torch.stack([z1, z2, z3])
        #embedding, _ = self.attention(inputs)
        #z, _ = self.attention(z)
        ##
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1


class GCN_DAE(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 non_linearity, normalization, mlp_h, mlp_epochs, gen_mode, sparse, mlp_act):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dgl(hidden_dim, nclasses))

        else:
            self.layers.append(GCNConv_dense(in_dim, hidden_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dense(hidden_dim, nclasses))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.k = k
        self.knn_metric = knn_metric
        self.i = i_
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.nnodes = features.shape[0]
        self.mlp_h = mlp_h
        self.mlp_epochs = mlp_epochs
        self.sparse = sparse

        if gen_mode == 0:
            self.graph_gen = FullParam(features, non_linearity, k, knn_metric, self.i, sparse).cuda()
        elif gen_mode == 1:
            self.graph_gen = MLP(2, features.shape[1], math.floor(math.sqrt(features.shape[1] * self.mlp_h)),
                                 self.mlp_h, mlp_epochs, k, knn_metric, self.non_linearity, self.i, self.sparse,
                                 mlp_act, features).cuda()
        elif gen_mode == 2:
            self.graph_gen = MLP_Diag(2, features.shape[1], k, knn_metric, self.non_linearity, self.i, sparse,
                                      mlp_act).cuda()

    def get_adj(self, h):
        Adj_ = self.graph_gen(h)
        if not self.sparse:
            Adj_ = symmetrize(Adj_)
            Adj_ = normalize(Adj_, self.normalization, self.sparse)
        return Adj_

    def forward(self, Adj_, x):  # x corresponds to masked_fearures
        #Adj_ = self.get_adj(features)
        if self.sparse:
            Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_
