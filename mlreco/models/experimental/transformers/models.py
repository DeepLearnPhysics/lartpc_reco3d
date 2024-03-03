import torch
import torch.nn as nn

from .layers import TransformerEncoderLayer, PositionWiseFFN
from .attention import SparseMultiHeadAttention, GraphTransformerLayer, _get_norm_layer, _get_act_layer


class VanillaTransformer(nn.Module):
    """Vanilla Transformer model for encoding N feature vectors into 
    another N feature vectors.
    
    """
    
    def __init__(self, vanilla_transformer):
        super(VanillaTransformer, self).__init__()
        
        self.process_model_config(**vanilla_transformer)
        
        self.tf_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            num_input = self.num_input if i == 0 else self.num_output
            self.tf_layers.append(
                TransformerEncoderLayer(num_input,
                        self.num_output,
                        self.num_hidden,
                        num_heads=self.num_heads,
                        leakiness=self.leakiness,
                        dropout=self.dropout,
                        norm_layer=self.norm_layer))
            
        self.node_predictor = nn.Linear(self.num_output, self.node_classes)
        
        
    def process_model_config(self, num_input, num_output, 
                             node_classes,
                             num_heads=8,
                             num_layers=6, 
                             num_hidden=128,
                             dropout=0.0, 
                             leakiness=0.0,
                             activation='relu', 
                             norm_layer='batch_norm',
                             use_bias=False, **kwargs):
        
        self.num_input = num_input
        self.num_output = num_output
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.leakiness = leakiness
        self.activation = activation
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.node_classes = node_classes
        

    def forward(self, x, edge_index=None, edge_attr=None, xbatch=None):
        """Forward pass of the model.

        Inputs
        ------
        x : torch.Tensor
            Input tensor of shape (N, num_input)

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (N, num_output)
        """

        node_features = x
        
        output = []
        
        for bidx in xbatch.unique():
            mask = xbatch == bidx
            node_input = x[mask]
            for layer in self.tf_layers:
                node_input = layer(node_input)
            output.append(node_input)
                
        output = torch.cat(output, dim=0)
            
        node_pred = self.node_predictor(output)
        edge_pred = edge_attr
            
        res = {
            'node_pred': [node_pred],
            'edge_pred': [edge_pred],
            'node_features': [node_features]
            }
        return res
    
    
class SPTransformer(nn.Module):
    """Sparse Transformer model for encoding N feature vectors into
    another N node feature vectors. The attention maps will be restricted
    by the input graph edge index information.

    """
    def __init__(self, sparse_transformer):
        super(SPTransformer, self).__init__()
        
        self.process_model_config(**sparse_transformer)
        
        self.tf_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            num_input = self.num_input if i == 0 else self.num_output
            self.tf_layers.append(
                SPTransformerEncoderLayer(num_input,
                        self.num_output,
                        self.num_hidden,
                        num_heads=self.num_heads,
                        leakiness=self.leakiness,
                        dropout=self.dropout,
                        norm_layer=self.norm_layer))
            
        self.node_predictor = nn.Linear(self.num_output, self.node_classes)
            
            
    def process_model_config(self, num_input, num_output, 
                             node_classes,
                             num_heads=8,
                             num_layers=6, 
                             num_hidden=128,
                             dropout=0.0, 
                             leakiness=0.0,
                             activation='relu', 
                             norm_layer='batch_norm',
                             use_bias=False, **kwargs):
        
        self.num_input = num_input
        self.num_output = num_output
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.leakiness = leakiness
        self.activation = activation
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.node_classes = node_classes
        
    def forward(self, x, edge_index, edge_attr=None, xbatch=None):
        
        node_features = x
        
        for layer in self.tf_layers:
            x = layer(x, edge_index)
            
        node_pred = self.node_predictor(x)
        edge_pred = edge_attr
            
        res = {
            'node_pred': [node_pred],
            'edge_pred': [edge_pred],
            'node_features': [node_features]
            }
            
        return res
    
    
class EdgeTransformer(nn.Module):
    
    def __init__(self, edge_transformer):
        super(EdgeTransformer, self).__init__()
        
        self.process_model_config(**edge_transformer)
        
        self.tf_layers = nn.ModuleList([])
        for i in range(self.num_layers):
            num_input = self.num_input if i == 0 else self.num_output
            edge_num_input = self.edge_num_input if i == 0 else self.num_output
            self.tf_layers.append(
                GraphTransformerEncoderLayer(num_input,
                        self.num_output,
                        self.num_heads,
                        self.num_hidden,
                        edge_num_input=edge_num_input,
                        leakiness=self.leakiness,
                        dropout=self.dropout,
                        norm_layer=self.norm_layer,
                        edge_include_mode=self.edge_include_mode))
            
        self.node_predictor = nn.Linear(self.num_output, self.node_classes)
        self.edge_predictor = nn.Linear(self.num_output, self.edge_classes)
        
    def process_model_config(self, num_input, num_output, 
                             edge_num_input,
                             node_classes,
                             edge_classes,
                             num_heads=8,
                             num_layers=6, 
                             num_hidden=128,
                             dropout=0.0, 
                             leakiness=0.0,
                             activation='relu', 
                             norm_layer='batch_norm',
                             use_bias=False,
                             edge_include_mode='prod',
                             **kwargs):
        
        self.num_input = num_input
        self.num_output = num_output
        self.edge_num_input = edge_num_input
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.leakiness = leakiness
        self.dropout = dropout
        self.activation = activation
        self.norm_layer = norm_layer
        self.use_bias = use_bias
        self.edge_include_mode = edge_include_mode
        
        self.node_classes = node_classes
        self.edge_classes = edge_classes
        
    def forward(self, x, edge_index, edge_attr, xbatch=None):
        
        node_features = x
            
        for layer in self.tf_layers:
            x, edge_attr = layer(x, edge_index, edge_attr, xbatch)
            
        node_pred = self.node_predictor(x)
        edge_pred = self.edge_predictor(edge_attr)
            
        res = {
            'node_pred': [node_pred],
            'edge_pred': [edge_pred],
            'node_features': [node_features]
            }
            
        return res
    
    
# --------------------------------------------------------------------

class SPTransformerEncoderLayer(nn.Module):
    
    def __init__(self, num_input, num_output,
                 num_hidden=128,
                 num_heads=8,
                 leakiness=0.0,
                 residual=True,
                 use_bias=False,
                 concat_attn=True,
                 dropout=0.0,
                 norm_layer='batch_norm'):
        super(SPTransformerEncoderLayer, self).__init__()
        
        assert num_output % num_heads == 0
        
        self.attention = SparseMultiHeadAttention(num_input, num_output,
                                                  num_heads,
                                                  dropout=dropout,
                                                  norm=norm_layer,
                                                  residual=residual,
                                                  use_bias=use_bias,
                                                  concat_attn=concat_attn)
        
        if num_input == num_output:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(num_input, num_output)
            
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = _get_norm_layer(norm_layer)(num_output)
        
        self.ffn = PositionWiseFFN(num_output, num_output,
                                   num_hidden,
                                   leakiness=leakiness,
                                   dropout=dropout,
                                   norm_layer=norm_layer)
        
        
    def forward(self, x, edge_index):
        
        _x = self.residual(x)
        x = self.attention(x, edge_index)
        
        x = self.dropout(x)
        x = self.norm1(x + _x)

        x = self.ffn(x)
        
        return x
    
    
class GraphTransformerEncoderLayer(nn.Module):
    
    def __init__(self, num_input, num_output, num_heads,
                 num_hidden, edge_num_input,
                 leakiness=0.0,
                 dropout=0.0,
                 norm_layer='batch_norm',
                 residual=True,
                 use_bias=False,
                 concat_attn=True,
                 edge_include_mode='prod'):
        super(GraphTransformerEncoderLayer, self).__init__()
        
        assert num_output % num_heads == 0
        
        self.attention = GraphTransformerLayer(num_input, num_output,
                                               num_heads,
                                               e_in_dim=edge_num_input,
                                               dropout=0.0,
                                               norm=norm_layer,
                                               residual=residual,
                                               use_bias=use_bias,
                                               concat_attn=concat_attn,
                                               edge_include_mode=edge_include_mode)
        
        if num_input == num_output:
            self.residual_x = nn.Identity()
        else:
            self.residual_x = nn.Linear(num_input, num_output)
            
        if edge_num_input == num_output:
            self.residual_e = nn.Identity()
        else:
            self.residual_e = nn.Linear(edge_num_input, num_output)
            
        self.norm_x = _get_norm_layer(norm_layer)(num_output)
        self.norm_e = _get_norm_layer(norm_layer)(num_output)
        
        self.ffn_node = PositionWiseFFN(num_output, num_output,
                                        num_hidden, 
                                        leakiness=leakiness,
                                        dropout=dropout,
                                        norm_layer=norm_layer)
        
        self.ffn_edge = PositionWiseFFN(num_output, num_output,
                                        num_hidden, 
                                        leakiness=leakiness,
                                        dropout=dropout,
                                        norm_layer=norm_layer)
        
        
    def forward(self, x, edge_index, edge_attr, xbatch=None):
        
        _x = self.residual_x(x)
        _e = self.residual_e(edge_attr)
        
        node_out, edge_out = self.attention(x, edge_index, edge_attr, xbatch)
        
        node_out = self.norm_x(node_out + _x)
        edge_out = self.norm_e(edge_out + _e)
        
        node_out = self.ffn_node(node_out)
        edge_out = self.ffn_edge(edge_out)
        
        return node_out, edge_out