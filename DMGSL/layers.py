import dgl.function as fn
import torch
import torch.nn as nn

EOS = 1e-10


class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)    # 两个张量矩阵相乘，在PyTorch中可以通过torch.matmul函数实现
        return output


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)    # 用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量

    def forward(self, x, g):
        with g.local_scope():   # 任何对节点或边的修改在脱离这个局部范围后将不会影响图中的原始特征值
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)


class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


class Diag(nn.Module):
    def __init__(self, input_size):
        super(Diag, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size))
        self.input_size = input_size

    def forward(self, input):
        hidden = input @ torch.diag(self.W)
        return hidden