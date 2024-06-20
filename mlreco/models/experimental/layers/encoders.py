import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label
from mlreco.utils.globals import *

from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from mlreco.models.experimental.layers.pointnet import PointNetEncoder
from mlreco.models.layers.gnn import node_encoder_construct, edge_encoder_construct

from collections import Counter

def encoder_construct(name):
    
    out = {
        'geometric': GeometricEncoder,
        'residual': SparseResidualEncoder,
        'pointnet': PointNetEncoder
    }
    
    return out[name]


def norm_construct(name):
    
    out = {
        'batchnorm': nn.BatchNorm1d,
        'layernorm': nn.LayerNorm
    }
    
    return out[name]


def compute_min_dist_to_point(points, values, other, r=5.0):
    
    dists = torch.norm(points - other, dim=1)
    min_dist = dists.min()
    min_index = torch.argmin(dists)
    
    closest_point = points[min_index]
    new_dists = torch.norm(points - closest_point, dim=1)
    
    dedx = (values[new_dists < r].sum() + 1e-6) / (new_dists[new_dists < r].max() + 1e-6)
    
    return min_dist, dedx


class GeometricEncoder(nn.Module):
    
    def __init__(self, cfg, name='geometric'):
        super(GeometricEncoder, self).__init__()
        
        self.split_col = cfg[name]['split_col']
        self.include_shower_feats = cfg[name].get('include_shower_feats', False)
        self.use_fourier_embeddings = cfg[name].get('use_fourier_embeddings', False)
        self.latent_size = cfg[name].get('latent_size', 16)
        
    @torch.no_grad()
    def encode_one(self, 
                   point_cloud: torch.Tensor, 
                   startpoint: torch.Tensor = None) -> torch.Tensor:
        """
        Encode a single point cloud.
        """
        voxels = point_cloud[:, COORD_COLS]
        values = point_cloud[:, VALUE_COL]
        
        size = torch.tensor([len(point_cloud)], 
                            dtype=voxels.dtype, device=voxels.device)
        center = voxels.mean(dim=0)
        x = voxels - center
        A = x.t().mm(x)
        
        w, v = torch.linalg.eigh(A, UPLO='U')
        dirwt = 1.0 - w[1] / w[2]
        B = A / w[2]
        
        v0 = v[:, 2]
        x0 = x.mv(v0)
        
        xp0 = x - torch.ger(x0, v0)
        np0 = torch.norm(xp0, dim=1)
        
        sc = torch.dot(x0, np0)
        if sc < 0:
            v0 = -v0
        
        v0 = dirwt * v0
        
        feats_v = torch.cat([center, B.flatten(), v0, w, size])
        
        if self.include_shower_feats:
            if startpoint is None:
                raise ValueError('Startpoints must be provided if include_shower_feats is True')
            min_dist, dedx = compute_min_dist_to_point(voxels, values, startpoint)
            feats_v = torch.cat([feats_v, torch.tensor([min_dist, dedx], 
                                                       dtype=voxels.dtype, 
                                                       device=voxels.device)])
            
        if feats_v.shape[0] != self.latent_size:
            msg = f'Latent size {self.latent_size} does not match '\
                f'the number of features in the encoder output {feats_v.shape[0]}'
            raise ValueError(msg)
            
        return feats_v
        
    @torch.no_grad()
    def forward(self, data: torch.Tensor, 
                startpoints: torch.Tensor = None) -> torch.Tensor:
        
        out = []
        clusts = form_clusters(data[:, :VALUE_COL].long(), 
                               column=self.split_col)
        
        for i, c in enumerate(clusts):
            if startpoints is None:
                v = self.encode_one(data[c])
            else:
                pt = startpoints[i][COORD_COLS]
                v = self.encode_one(data[c], pt)
            out.append(v)
            
        out = torch.stack(out, dim=0)
        return out
    
    
class MixedEncoder(nn.Module):
    
    def __init__(self, cfg, name='mixed_encoder'):
        super(MixedEncoder, self).__init__()
        
        self.encoder1_name = cfg[name]['encoder1_name']
        self.encoder2_name = cfg[name]['encoder2_name']
        
        assert self.encoder1_name != self.encoder2_name
        
        norm = cfg[name]['norm']
        norm_kwargs = cfg[name].get('norm_kwargs', {})
        
        self.encoder1 = encoder_construct(self.encoder1_name)(cfg)
        self.encoder2 = encoder_construct(self.encoder2_name)(cfg)
        
        self.latent_size = self.encoder1.latent_size + self.encoder2.latent_size
        
        self.norm = norm_construct(norm)(self.latent_size, **norm_kwargs)
        
    def forward(self, data: torch.Tensor, startpoints=None) -> torch.Tensor:
        
        out1 = self.encoder1(data, startpoints)
        out2 = self.encoder2(data, startpoints)
        
        out = torch.cat([out1, out2], dim=1)
        return self.norm(out)
        
        
class MixedGNNEncoder(nn.Module):
    
    def __init__(self, cfg, **kwargs):
        super(MixedGNNEncoder, self).__init__()
        self.geo_encoder = node_encoder_construct(cfg, model_name='geo_encoder', **kwargs)
        self.param_encoder = node_encoder_construct(cfg, model_name='net_encoder', **kwargs)
        self.num_features = self.geo_encoder.num_features + self.param_encoder.num_features
        
        self.mix_norm = nn.BatchNorm1d(self.num_features)
        
        print(f"Geo({self.geo_encoder.num_features}) + {self.param_encoder}({self.param_encoder.num_features}) = {self.num_features}")
        
    def forward(self, data, clusts=None):
        
        feats_geo = self.geo_encoder(data, clusts)
        feats_cnn = self.param_encoder(data, clusts)
        
        # pid_labels = get_cluster_label(data, clusts, column=PID_COL).astype(int)
        
        # feats_geo = feats_geo[pid_labels != -1]
        
        out = torch.cat([feats_geo, feats_cnn], dim=1)

        return out