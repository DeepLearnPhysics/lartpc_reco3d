import torch
import torch.nn as nn
import MinkowskiEngine as ME

from functools import partial
from torch_cluster import fps, knn


class FPSKNNGroup(nn.Module):
    
    def __init__(self, ratio=0.1, k=5):
        super(FPSKNNGroup, self).__init__()
        self.fps = partial(fps, ratio=ratio)
        self.knn = partial(knn, k=k)
        
    def forward(self, x, pos, batch):
        """
            x: (N, F)
            pos: (N, C)
            batch: (N, )
            
            out: (N, F_emb), F_emb >= F
        """
        fps_idx = self.fps(pos, batch=batch)
        fps_batch = batch[fps_idx]
        centroids = pos[fps_idx]
        groups_idx = self.knn(pos, centroids, batch_x=batch, batch_y=fps_batch)
        groups = pos[groups_idx[1]]
        return centroids, groups
    
