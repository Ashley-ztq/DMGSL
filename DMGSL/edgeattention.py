import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播.
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


# 加性模型
class Edgeattention(nn.Module):
    def __init__(self, input_size, output_size):
        # q、k、v的维度，seq_len每句话中词的数量
        super(Edgeattention, self).__init__()
        #self.linear_v = nn.Linear(v_size, seq_len)
        self.linear_W = nn.Linear(input_size, output_size).cuda()
        #self.linear_U = nn.Linear(q_size, q_size)
        self.q = torch.randn(output_size, 1).cuda()
        #self.bias = torch.randn(output_size)
        #self.tanh = nn.Tanh()
        self.tanh = nn.Tanh()
        #self.layer_norm = nn.LayerNorm(input_size)
        self.layer_norm = nn.LayerNorm(output_size)

    #def forward(self, query, key, value, dropout=None):
        #key = self.linear_W(key)
        #print(key.shape)
        #query = self.linear_U(query)
        #print(query.shape)
        #k_q = self.tanh(query + key)
        #print(k_q.shape)
        #alpha = self.linear_v(k_q)
        #print(alpha.shape)
        #alpha = F.softmax(alpha, dim=-1)
        #print(alpha.shape)
        ##out = torch.bmm(alpha, value)
        #out = torch.tensordot(alpha, value, dims=1)
        #return out, alpha

    def forward(self, inputs):
        #print(inputs.shape)
        multi_val = self.linear_W(inputs)
        #print('1', multi_val.shape)
        multi_val_1 = self.tanh(multi_val)
        #print('2',multi_val)
        #print('q', self.q)
        #self.q.to(device)
        #multi_val.to(device)
        multi_val = torch.tensordot(multi_val_1, self.q, dims=1)
        #print('3', multi_val)
        beta = F.softmax(multi_val_1, dim=0)
        #print('4', beta)
        out = torch.sum(multi_val * beta, 0)
        out = self.layer_norm(out)
        #print(out.shape)
        return out, beta


#attention_1 = Edgeattention(3, 4)
#inputs = torch.randn((4,2,3))  # 可以理解为有8句话，每句话有10个词，每个词用100维的向量来表示
#out, attn = attention_1(inputs)
#print(out.shape)
#print(attn.shape)


class LSTMLearnerLayer(nn.Module):
    def __init__(self, input_dim, num_time_steps, act=nn.Sigmoid, name=None):
        super(LSTMLearnerLayer, self).__init__()

        self.input_dim = input_dim
        self.num_time_steps = num_time_steps
        self.act = act
        self.tanh = nn.Tanh()
        #self.tanh = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.lstm_weights_gx = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_gh = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_ix = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_ih = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_fx = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_fh = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_ox = nn.Linear(input_dim, input_dim).cuda()
        self.lstm_weights_oh = nn.Linear(input_dim, input_dim).cuda()

    def forward(self, inputss):
        #print(inputss.shape)
        graphs = inputss
        #print(graphs.shape)
        state_h_t = torch.ones_like(graphs[0, :, :])
        state_s_t = torch.ones_like(graphs[0, :, :])
        #print('state_h_t', state_h_t.shape)
        outputs = []
        for idx in range(0, self.num_time_steps):
            inputs = graphs[idx, :, :]
            gate_g_ = self.lstm_weights_gx(inputs)
            gate_g_ += self.lstm_weights_gh(state_h_t)
            gate_g = self.tanh(gate_g_)

            gate_i_ = self.lstm_weights_ix(inputs)
            gate_i_ += self.lstm_weights_ih(state_h_t)
            gate_i = self.sigmoid(gate_i_)

            gate_f_ = self.lstm_weights_fx(inputs)
            gate_f_ += self.lstm_weights_fh(state_h_t)
            gate_f = self.sigmoid(gate_f_)

            gate_o_ = self.lstm_weights_ox(inputs)
            gate_o_ += self.lstm_weights_oh(state_h_t)
            gate_o = self.sigmoid(gate_o_)

            #print('inputs', inputs.shape)
            #print(inputs.shape)
            #print(self.lstm_weights_gx.weight.shape)
            #gate_g_ = torch.transpose(torch.matmul(self.lstm_weights_gx.weight, inputs), dim0=0, dim1=1) + self.lstm_weights_gx.bias
            #gate_g_ += torch.transpose(torch.matmul(self.lstm_weights_gh.weight, state_h_t), dim0=0, dim1=1) + self.lstm_weights_gh.bias
            ########################
            #bn = torch.nn.BatchNorm1d(gate_g_.shape[1]).cuda()
            #gate_g_ = bn(gate_g_)
            #######################
            #gate_g = torch.transpose(self.tanh(gate_g_), dim0=0, dim1=1)
            #print(gate_g.shape)

            #gate_i_ = torch.transpose(torch.matmul(self.lstm_weights_ix.weight, inputs), dim0=0, dim1=1) + self.lstm_weights_ix.bias
            #gate_i_ += torch.transpose(torch.matmul(self.lstm_weights_ih.weight, state_h_t), dim0=0, dim1=1) + self.lstm_weights_ih.bias
            ####################
            #bn = torch.nn.BatchNorm1d(gate_i_.shape[1]).cuda()
            #gate_i_ = bn(gate_i_)
            #####################
            #gate_i = torch.transpose(self.sigmoid(gate_i_), dim0=0, dim1=1)
            #print(gate_i.shape)

            #gate_f_ = torch.transpose(torch.matmul(self.lstm_weights_fx.weight, inputs), dim0=0, dim1=1) + self.lstm_weights_fx.bias
            #gate_f_ += torch.transpose(torch.matmul(self.lstm_weights_fh.weight, state_h_t), dim0=0, dim1=1) + self.lstm_weights_fh.bias
            ####################
            #bn = torch.nn.BatchNorm1d(gate_f_.shape[1]).cuda()
            #gate_f_ = bn(gate_f_)
            ####################
            #gate_f = torch.transpose(self.sigmoid(gate_f_), dim0=0, dim1=1)
            #print(gate_f.shape)

            #gate_o_ = torch.transpose(torch.matmul(self.lstm_weights_ox.weight, inputs), dim0=0, dim1=1) + self.lstm_weights_ox.bias
            #gate_o_ += torch.transpose(torch.matmul(self.lstm_weights_oh.weight, inputs), dim0=0, dim1=1) + self.lstm_weights_oh.bias
            ###################
            #bn = torch.nn.BatchNorm1d(gate_o_.shape[1]).cuda()
            #gate_o_ = bn(gate_o_)
            ####################
            #gate_o = torch.transpose(self.sigmoid(gate_o_), dim0=0, dim1=1)
            #print(gate_o.shape)
            #print(state_s_t.shape)

            #gate_g_ = self.lstm_weights_gx(inputs)
            #gate_g_ += self.lstm_weights_gh(state_h_t)
            #gate_g = self.tanh(gate_g_)
            #print('gate_g', gate_g.shape)

            #gate_i_ = self.lstm_weights_ix(inputs)
            #gate_i_ += self.lstm_weights_ih(state_h_t)
            #gate_i = self.sigmoid(gate_i_)
        #    print(gate_i.shape)

            #gate_f_ = self.lstm_weights_fx(inputs)
            #gate_f_ += self.lstm_weights_fh(state_h_t)
            #gate_f = self.sigmoid(gate_f_)
        #    print(gate_f.shape)

            #gate_o_ = self.lstm_weights_ox(inputs)
            #ate_o_ += self.lstm_weights_oh(state_h_t)
            #gate_o = self.sigmoid(gate_o_)
            #print(gate_g.shape)
            #print(state_s_t.shape)
            state_s = torch.multiply(gate_g, gate_i) + torch.multiply(state_s_t, gate_f)
            #print('state_s', state_s.shape)
            ################
            #bn = torch.nn.BatchNorm1d(state_s.shape[1]).cuda()
            #state_s = bn(state_s)
            ################
            state_h = torch.multiply(self.tanh(state_s), gate_o)
            #print('state_h', state_h.shape)

            state_h_t = state_h
            state_s_t = state_s
            outputs.append(state_h)

        outputs = torch.stack(outputs)
        outputs = self.layer_norm(outputs + graphs)
        #outputs = outputs.permute(0,2,1)
        return outputs


class TemporalAttentionLayer(nn.Module):
    def __init__(self, input_dim, n_heads, num_time_steps, attn_drop, residual=False, bias=True,
                 use_position_embedding=True):
        super(TemporalAttentionLayer, self).__init__()

        self.bias = bias
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.attn_drop = attn_drop
        self.attn_wts_means = []
        self.attn_wts_vars = []
        self.residual = residual
        self.input_dim = input_dim
        self.dim_per_head = input_dim // n_heads
        self.dot_product_attention = ScaledDotProductAttention(0.0)

        if use_position_embedding:
            self.position_embeddings = nn.Linear(input_dim, self.num_time_steps)

        self.Q_embedding = nn.Linear(input_dim, input_dim)
        self.K_embedding = nn.Linear(input_dim, input_dim)
        self.V_embedding = nn.Linear(input_dim, input_dim)
        self.linear_final = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(self.attn_drop)
        self.dropout2 = nn.Dropout(0.0)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.softmax = nn.Softmax()

    def forward(self, inputs, attn_mask=None):

        inputs_reshaped = inputs
        pe = torch.zeros(self.num_time_steps, self.input_dim).cuda()
        position = torch.arange(0, self.num_time_steps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2).float() * (-math.log(10000.0) / self.input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        inputs_temporal = inputs_reshaped + pe[:inputs_reshaped.size(0), :]
        inputs_temporal = self.layer_norm(inputs_temporal+inputs_reshaped)
        #inputs_temporal = inputs_reshaped
        #print(inputs_temporal.shape)
        '''
        # 1: Add position embeddings to input
        position_inputs = torch.tile(torch.arange(0, self.num_time_steps).unsqueeze(0),
                                     [inputs_reshaped.shape[1], 1]).to(device)  # 例：(3,2) tile通过重复 input 的元素构造张量。的 reps 参数指定在每个维度重复次数。
        # print(position_inputs.shape)
        position_embeddings = torch.embedding(self.position_embeddings.weight, position_inputs).to(device).permute(1, 0, 2)  # 例：(3,4,2)
        # print(self.position_embeddings.weight.shape)
        #print(position_embeddings.shape)
        inputs_temporal = inputs_reshaped + position_embeddings  # 例：(3,4,2)
        #inputs_temporal = inputs_reshaped
        
        
        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(inputs_temporal, self.Q_embedding.weight, dims=([2], [0]))  # 例：(3,4,2)
        # print("q", q.shape)
        k = torch.tensordot(inputs_temporal, self.K_embedding.weight, dims=([2], [0]))  # 例：(3,4,2)
        # print(k)
        v = torch.tensordot(inputs_temporal, self.V_embedding.weight, dims=([2], [0]))  # 例：(3,4,2)
        # print(v)

        
        #3
        residual = inputs_temporal
        key = k.view(self.num_time_steps * self.n_heads, -1, self.dim_per_head)
        #print(key.size(0), key.size(1), key.size(2))
        value = v.view(self.num_time_steps * self.n_heads, -1, self.dim_per_head)
        query = q.view(self.num_time_steps * self.n_heads, -1, self.dim_per_head)
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(self.num_time_steps, -1, self.dim_per_head * self.n_heads)
        #print(context.size(0), context.size(1), context.size(2))
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        output = output.permute(1, 0, 2)
        return output
        '''
        '''
        # 3
        outputs = torch.matmul(q.permute(1, 0, 2), k.permute(1, 2, 0))
        outputs = outputs / (self.input_dim ** 0.5)

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0, :, :]).to(device)  # 例：(4,4)
        # print(diag_val)
        diag_val = diag_val.cpu().detach().numpy()
        tril = np.tril(diag_val)
        row, col = np.nonzero(tril)
        values = tril[row, col]
        csr_a = csr_matrix((values, (row, col)), shape=diag_val.shape)
        tril = csr_a.todense()
        # tril = torch.tril(diag_val).to_dense()
        # tril = tril.todense()
        tril = torch.tensor(tril).to(device)
        # print(tril)
        masks = torch.tile(tril.unsqueeze(0), [outputs.shape[0], 1, 1])
        # print(type(masks))
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        # print(type(padding))
        outputs = torch.where(torch.eq(masks, 0), padding, outputs)  # 按照一定的规则合并两个tensor类型
        # print(type(outputs))
        outputs = F.softmax(outputs, dim=1)
        # print("output_4", outputs.shape)
        # self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = F.dropout(outputs, p=self.attn_drop)
        # print(outputs)
        outputs = torch.matmul(outputs, v.permute(1, 0, 2))
        # only return the last snapshot embeddings
        #final_outputs = torch.squeeze(outputs[:, outputs.shape[1] - 1, :])
        # print("out_5", final_outputs.shape)
        return outputs
        '''
        '''
        inputs_reshaped = inputs    # 例：(2,3,4)
        #print(inputs.shape)
        # 1: Add position embeddings to input
        #position_inputs = torch.tile(torch.arange(0, self.num_time_steps).unsqueeze(0), [inputs_reshaped.shape[1], 1]).to(device)    # 例：(3,2) tile通过重复 input 的元素构造张量。的 reps 参数指定在每个维度重复次数。
        #print(position_inputs.shape)
        #position_embeddings = torch.embedding(self.position_embeddings.weight, position_inputs).to(device).permute(1,0,2) # 例：(3,4,2)
        #print(self.position_embeddings.weight.shape)
        #print(position_embeddings.shape)
        #inputs_temporal = inputs_reshaped + position_embeddings # 例：(3,4,2)
        #print("inputs_temporal", inputs_temporal.shape)

        inputs_temporal = inputs_reshaped
        '''
        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(inputs_temporal, self.Q_embedding.weight, dims=([2],[0]))   # 例：(3,4,2)
        #print("q", q.shape)
        k = torch.tensordot(inputs_temporal, self.K_embedding.weight, dims=([2],[0]))   # 例：(3,4,2)
        #print(k)
        v = torch.tensordot(inputs_temporal, self.V_embedding.weight, dims=([2],[0]))   # 例：(3,4,2)
        #print(v)

        # 3: Split, concat and scale
        q_ = torch.cat(torch.split(q, int(q.shape[2]/self.n_heads), dim=2), dim=0).permute(1, 0, 2)  # 例：(3,4,2)
        #print("q_", q_.shape)
        k_ = torch.cat(torch.split(k, int(q.shape[2]/self.n_heads), dim=2), dim=0).permute(1, 0, 2)  # 例：(3,4,2)
        #print(k_)
        v_ = torch.cat(torch.split(v, int(q.shape[2]/self.n_heads), dim=2), dim=0).permute(1, 0, 2)  # 例：(3,4,2)
        #print("v_",v_)

        #outputs = torch.tensordot(q_, k_.permute(0, 2, 1), dims=([2], [1]))  # 例：(3,4,4)
        outputs = torch.matmul(q_, k_.permute(0, 2, 1)) # 例：(3,4,4)
        #print(outputs)
        outputs = outputs / (self.input_dim ** 0.5)     # 例：(3,4,4)  **表示乘幂
        #print("output_3", outputs.shape)

        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0, :, :])   # 例：(4,4)
        #print(diag_val)
        diag_val = diag_val.cpu().detach().numpy()
        tril = np.tril(diag_val)
        row, col = np.nonzero(tril)
        values = tril[row, col]
        csr_a = csr_matrix((values, (row, col)), shape=diag_val.shape)
        tril = csr_a.todense()
        #tril = torch.tril(diag_val).to_dense()
        #tril = tril.todense()
        tril = torch.tensor(tril).to(device)
        #print(tril)
        masks = torch.tile(tril.unsqueeze(0), [outputs.shape[0], 1, 1])
        #print(type(masks))
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        #print(type(padding))
        outputs = torch.where(torch.eq(masks, 0), padding, outputs) # 按照一定的规则合并两个tensor类型
        #print(type(outputs))
        outputs = self.softmax(outputs)
        #print("output_4", outputs.shape)
        #self.attn_wts_all = outputs

        # 5: Dropout on attention weights.
        outputs = self.dropout1(outputs)
        #print(outputs)
        ############################
        #split_outputs = []
        v_split = torch.zeros_like(v_)
        for i in range(0, self.n_heads):
            #print("1", v_split.shape)
            v_split[:, self.num_time_steps*i:self.num_time_steps*(i+1), self.num_time_steps*i:self.num_time_steps*(i+1)] = v_[:, self.num_time_steps*i:self.num_time_steps*(i+1), self.num_time_steps*i:self.num_time_steps*(i+1)]
            #print("2", v_split.shape)
            #print("3", outputs.shape)
            outputs_ = torch.matmul(outputs, v_split)
            #print("4", outputs_.shape)
            #split_outputs.append(outputs_)
        outputs = torch.cat(torch.split(outputs_, int(outputs_.shape[1]/self.n_heads), dim=1), dim=-1)
        #print(outputs.shape)
        #############################
        #outputs = torch.matmul(outputs, v_)
        #print("output_51", outputs.shape)
        #split_outputs = torch.split(outputs, int(outputs.shape[1]/self.n_heads), dim=1)
        #print(split_outputs)
        #outputs = torch.cat(split_outputs, dim=-1)
        residual = inputs_reshaped
        #residual = inputs_reshaped
        output = self.linear_final(outputs)
        output = self.dropout2(output)
        residual = residual.permute(1, 0, 2)
        output = self.layer_norm(residual + output)
        #print("output_5", outputs.shape)
        # only return the last snapshot embeddings
        #final_outputs = torch.squeeze(outputs[:, outputs.shape[1]-1, :])
        #print("out_5", final_outputs.shape)
        return output



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        final_outputs = torch.squeeze(output[:, output.shape[1] - 1, :])
        #print(final_outputs.shape)
        return final_outputs

#model = TemporalAttentionLayer(2, 5, 4, 0.0)
#output = model(input)

