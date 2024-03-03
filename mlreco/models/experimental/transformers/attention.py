import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm


def _get_norm_layer(name):
    norm_dict = {
        'batch_norm': nn.BatchNorm1d,
        'layer_norm': nn.LayerNorm
    }
    return norm_dict[name]

def _get_act_layer(name):
    act_dict = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU
    }
    return act_dict[name]


class SparseMultiHeadAttention(MessagePassing):
    """
    A multi-head attention layer that limits the
    attention map with the input graph connectivity information.
    """    
    _MIN_CLAMP = -5
    _MAX_CLAMP = 5
    
    def __init__(self, h_in_dim, out_dim, num_heads, 
                 dropout=0.0, norm='layer_norm',
                 residual=True, use_bias=False, concat_attn=True):
        super(SparseMultiHeadAttention, self).__init__(aggr='add')
        
        # Node In
        self.h_in_dim = h_in_dim
        self.out_dim = out_dim
        # Edge In
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_fn = _get_norm_layer(norm)
        self.residual = residual
        self.use_bias = use_bias
        self.concat_attn = concat_attn
        
        # Architecture Definitions
        
        self.norm1_h = self.norm_fn(h_in_dim)
        
        # Node Q, K, V
        self.Q = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        self.K = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        self.V = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        
        self.O_h = nn.Linear(self.out_dim * num_heads, self.out_dim, bias=use_bias)
        
    def forward(self, node_features, edge_index):
        
        # x (nodes, N x D_node ), e (edge, E x D_edge)
        h = self.norm1_h(node_features)
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        V_mod = self.propagate(edge_index, query=Q_h, key=K_h, value=V_h)

        if self.concat_attn:
            V_mod = V_mod.view(-1, self.num_heads * self.out_dim)
        else:
            V_mod = V_mod.mean(dim=1)
        
        out_node = self.O_h(V_mod)
        
        return out_node
        
        
    def message(self, query_i, key_j, value_j, index, ptr, size_i):
        
        q = query_i.view(-1, self.num_heads, self.out_dim)
        k = key_j.view(-1, self.num_heads, self.out_dim)
        
        # Scaled Dot Product
        A = (q * k).sum(dim=-1) / math.sqrt(self.out_dim)# E x num_heads
        # Clamping for numerial stability
        h = torch.clamp(A, min=self._MIN_CLAMP, max=self._MAX_CLAMP)
        # Compute and save Attention Map
        alpha = softmax(h, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Output Nodes features are attention map applied value vectors
        out_nodes = value_j.view(-1, self.num_heads, self.out_dim)
        out_nodes = out_nodes * alpha.view(-1, self.num_heads, 1)
        out_nodes = out_nodes.view(-1, self.num_heads * self.out_dim)
        
        return out_nodes
    
    

class GraphTransformerLayer(MessagePassing):
    """
    A Transformer multi-head attention layer that limits the
    attention map with the input graph connectivity information.
    
    The implementation follows closely that of:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/transformer_conv.html
    
    This model is inspired from the following papers:
    https://arxiv.org/pdf/2012.09699.pdf
    https://arxiv.org/pdf/2108.03348.pdf
    https://arxiv.org/pdf/2009.03509.pdf
    """
    
    _MIN_CLAMP = -5
    _MAX_CLAMP = 5
    
    def __init__(self, h_in_dim, out_dim, num_heads, 
                 e_in_dim=None, dropout=0.0, norm='layer_norm', 
                 residual=True, use_bias=False, concat_attn=True, 
                 edge_include_mode='add'):
        super(GraphTransformerLayer, self).__init__(aggr='add')
        
        # Node In
        self.h_in_dim = h_in_dim
        self.out_dim = out_dim
        # Edge In
        if e_in_dim is not None:
            self.e_in_dim = e_in_dim
        else:
            self.e_in_dim = h_in_dim
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_fn = _get_norm_layer(norm)
        self.residual = residual
        self.use_bias = use_bias
        self.concat_attn = concat_attn
        self.edge_include_mode = edge_include_mode
        
        # Architecture Definitions
        
        self.norm1_h = self.norm_fn(h_in_dim)
        self.norm1_e = self.norm_fn(e_in_dim)
        
        # Node Q, K, V
        self.Q = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        self.K = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        self.V = nn.Linear(h_in_dim, num_heads * self.out_dim, bias=use_bias)
        
        self.O_h = nn.Linear(self.out_dim * num_heads, self.out_dim, bias=use_bias)
        self.O_e = nn.Linear(self.out_dim * num_heads, self.out_dim, bias=use_bias)
        
        self.E = nn.Linear(self.e_in_dim, num_heads * self.out_dim, bias=use_bias)
        
        
    def forward(self, node_features, edge_index, edge_features, xbatch=None):
        
        # x (nodes, N x D_node ), e (edge, E x D_edge)
        h = self.norm1_h(node_features)
        e = self.norm1_e(edge_features)
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        
        V_mod = self.propagate(edge_index, query=Q_h, key=K_h, value=V_h, edge_attr=edge_features)
        
        alpha = self._alpha # Attention map (E x H)
        qk_edge = self._qk_edge # Q * K * proj_e (E x H x out_dim)

        if self.concat_attn:
            V_mod = V_mod.view(-1, self.num_heads * self.out_dim)
            E_mod = qk_edge.view(-1, self.num_heads * self.out_dim)
        else:
            V_mod = V_mod.mean(dim=1)
            E_mod = qk_edge.mean(dim=1)
        
        out_node = self.O_h(V_mod)
        out_edge = self.O_e(E_mod)
        
        return out_node, out_edge
        
        
    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        
        E_e = self.E(edge_attr).view(-1, self.num_heads, self.out_dim)
        
        q = query_i.view(-1, self.num_heads, self.out_dim)
        k = key_j.view(-1, self.num_heads, self.out_dim)
        
        if self.edge_include_mode == 'add':
            qk_edge = q * (k + E_e)
        elif self.edge_include_mode == 'prod':
            qk_edge = q * k * E_e
        else:
            raise ValueError(f"Edge feature inclusion mode {self.edge_include_mode} not supported.")
        self._qk_edge = qk_edge
        # Scaled Dot Product
        A = (qk_edge).sum(dim=-1) / math.sqrt(self.out_dim)# E x num_heads
        # Clamping for numerial stability
        h = torch.clamp(A, min=self._MIN_CLAMP, max=self._MAX_CLAMP)
        # Compute and save Attention Map
        alpha = softmax(h, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Output Nodes features are attention map applied value vectors
        out_nodes = value_j.view(-1, self.num_heads, self.out_dim)
        out_nodes = out_nodes * alpha.view(-1, self.num_heads, 1)
        out_nodes = out_nodes.view(-1, self.num_heads * self.out_dim)
        
        return out_nodes
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_channels={self.h_in_dim}, '
                f'out_channels={self.out_dim}, heads={self.num_heads})')
        
        
class TransformerConvLayer(nn.Module):
    """Transformer layer for graph data.
    
    This is a generalization of the Transformer layer for graph-structured
    data. The layer is based on the TransformerConv module from PyTorch and
    adds a feed-forward network (FFN) and residual connections. Design is
    taken from https://github.com/graphdeeplearning/graphtransformer with 
    some modifications.

    Parameters
    ----------
    nn : _type_
        _description_
    """
    
    def __init__(self, in_dim, out_dim, num_heads, 
                 edge_dim=None,
                 dropout=0.0, 
                 layer_norm=False, 
                 batch_norm=True, 
                 residual=True, 
                 use_bias=False):
        super(TransformerConvLayer, self).__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        
        self.attention = TransformerConv(in_dim, out_dim, 
                                         heads=num_heads, 
                                         dropout=dropout,
                                         edge_dim=edge_dim)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.norm1_h = LayerNorm(out_dim)
            self.norm1_e = LayerNorm(out_dim)
            
        if self.batch_norm:
            self.norm1_h = BatchNorm(out_dim)
            self.norm1_e = BatchNorm(out_dim)
            
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)
        
    def forward(self, x, edge_index, edge_attr):
        
        h_in1 = x
        e_in1 = edge_attr
        
        h = self.attention(x, edge_index, edge_attr)
        
        h = F.dropout(h, p=self.dropout, training=self.training)
        e = F.dropout(e, p=self.dropout, training=self.training)
        
        h = self.O_h(h)
        e = self.O_e(e)
        
        if self.residual:
            h = h_in1 + h
            e = e_in1 + e
            
        h = self.norm1_h(h)
        e = self.norm1_e(e)
            
        h_in2 = h
        e_in2 = e
            
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)
        
        if self.residual:
            h = h_in2 + h
            e = e_in2 + e
        
        h = self.norm2_h(h) # Node Features
        e = self.norm2_e(e) # Edge Features
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual)