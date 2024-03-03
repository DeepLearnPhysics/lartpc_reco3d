import random
import torch
import torch.nn as nn
import numpy as np

from mlreco.models.layers.common.dbscan import DBSCANFragmenter
from mlreco.models.layers.gnn import (gnn_model_construct, 
                                      node_encoder_construct, 
                                      edge_encoder_construct, 
                                      node_loss_construct, 
                                      edge_loss_construct)

from mlreco.utils.globals import *
from mlreco.utils.gnn.data import merge_batch, split_clusts, split_edge_index
from mlreco.utils.gnn.cluster import (form_clusters, 
                                      get_cluster_batch, 
                                      get_cluster_label, 
                                      get_cluster_primary_label, 
                                      get_cluster_points_label, 
                                      get_cluster_directions, 
                                      get_cluster_dedxs)
from mlreco.utils.gnn.network import (complete_graph, 
                                      delaunay_graph, 
                                      mst_graph, 
                                      bipartite_graph, 
                                      inter_cluster_distance, 
                                      knn_graph, 
                                      restrict_graph)


class ParticleNet(nn.Module):
    
    RETURNS = {
        'batch_ids': ['tensor'],
        'clusts' : ['index_list', ['input_data', 'batch_ids'], True],
        'node_features': ['tensor', 'batch_ids', True],
        'node_pred': ['tensor', 'batch_ids', True],
        'start_points': ['tensor', 'batch_ids', False, True],
        'end_points': ['tensor', 'batch_ids', False, True],
        'edge_features': ['edge_tensor', ['edge_index', 'batch_ids'], True],
        'edge_index': ['edge_tensor', ['edge_index', 'batch_ids'], True],
        'edge_pred': ['edge_tensor', ['edge_index', 'batch_ids'], True]
    }
    
    def __init__(self, cfg, name='particlenet'):
        super(ParticleNet, self).__init__()
        
        self.process_model_config(cfg, name)
        
        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg[name])
        self.edge_encoder = edge_encoder_construct(cfg[name])

        # Construct the GNN
        self.gnn_model = gnn_model_construct(cfg[name])
        
        
    def process_model_config(self, cfg, name='particlenet'):
        
        # Get the chain input parameters
        base_config = cfg[name].get('base', {})
        self.name = name
        self.batch_index = BATCH_COL
        self.coords_index = COORD_COLS

        # Choose what type of node to use
        self.source_col       = base_config.get('source_col', 5)
        self.target_col       = base_config.get('target_col', 6)
        self.node_type        = base_config.get('node_type', -1)
        self.node_min_size    = base_config.get('node_min_size', -1)
        self.add_points       = base_config.get('add_points', False)
        self.add_local_dirs   = base_config.get('add_local_dirs', False)
        self.dir_max_dist     = base_config.get('dir_max_dist', 5)
        self.opt_dir_max_dist = self.dir_max_dist == 'optimize'
        self.add_local_dedxs  = base_config.get('add_local_dedxs', False)
        self.dedx_max_dist    = base_config.get('dedx_max_dist', 5)
        self.break_clusters   = base_config.get('break_clusters', False)
        self.shuffle_clusters = base_config.get('shuffle_clusters', False)

        # *Deprecated* but kept for backward compatibility:
        if 'add_start_point'    in base_config: self.add_points = base_config['add_start_point']
        if 'add_start_dir'      in base_config: self.add_local_dirs = 'start' if base_config['add_start_dir'] else False
        if 'add_start_dedx'     in base_config: self.add_local_dedxs = 'start' if base_config['add_start_dedx'] else False
        if 'start_dir_max_dist' in base_config: self.dir_max_dist = self.dedx_max_dist = base_config['start_dir_max_dist']
        if 'start_dir_opt'      in base_config: self.opt_dir_max_dist = base_config['start_dir_opt']

        # Interpret node type as list of classes to cluster, -1 means all classes
        if isinstance(self.node_type, int): self.node_type = [self.node_type]

        # Choose what type of network to use
        self.network = base_config.get('network', 'complete')
        self.edge_max_dist = base_config.get('edge_max_dist', -1)
        self.edge_dist_metric = base_config.get('edge_dist_metric', 'voxel')
        self.edge_dist_algorithm = base_config.get('edge_dist_algorithm', 'brute')
        self.edge_knn_k = base_config.get('edge_knn_k', 5)
        self.edge_max_count = base_config.get('edge_max_count', 2e6)

        # Turn the edge_max_dist value into a matrix
        if not isinstance(self.edge_max_dist, list): self.edge_max_dist = [self.edge_max_dist]
        mat_size = int((np.sqrt(8*len(self.edge_max_dist)+1)-1)/2)
        max_dist_mat = np.zeros((mat_size, mat_size), dtype=float)
        max_dist_mat[np.triu_indices(mat_size)] = self.edge_max_dist
        max_dist_mat += max_dist_mat.T - np.diag(np.diag(max_dist_mat))
        self.edge_max_dist = max_dist_mat

        # If requested, merge images together within the batch
        self.merge_batch = base_config.get('merge_batch', False)
        self.merge_batch_mode = base_config.get('merge_batch_mode', 'const')
        self.merge_batch_size = base_config.get('merge_batch_size', 2)
        if self.merge_batch_mode not in ['const', 'fluc']:
            raise ValueError('Batch merging mode not supported, must be one of const or fluc')
        self.merge_batch_fluc = self.merge_batch_mode == 'fluc'

        # If requested, use DBSCAN to form clusters from semantics
        if 'dbscan' in cfg[name]:
            cfg[name]['dbscan']['cluster_classes'] = self.node_type if self.node_type[0] > -1 else [0,1,2,3]
            cfg[name]['dbscan']['min_size']        = self.node_min_size
            self.dbscan = DBSCANFragmenter(cfg[name], name='dbscan',
                                            batch_col=self.batch_index,
                                            coords_col=self.coords_index)
            
        
    def forward(self, data, clusts=None, 
                groups=None, points=None, extra_feats=None, batch_size=None):
        
        cluster_data = data[0]
        if len(data) > 1: 
            particles = data[1]
        else:
            raise ValueError('ParticleNet requires a list of particles as input')
        result = {}

        # Form list of list of voxel indices, one list per cluster in the requested class
        if clusts is None:
            if hasattr(self, 'dbscan'):
                clusts = self.dbscan(cluster_data, points=particles if len(data) > 1 else None)
            else:
                clusts = form_clusters(cluster_data.detach().cpu().numpy(),
                                       self.node_min_size,
                                       self.source_col,
                                       cluster_classes=self.node_type)
                if self.break_clusters:
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps=1.1, min_samples=1, metric='chebyshev')
                    broken_clusts = []
                    for c in clusts:
                        labels = dbscan.fit(cluster_data[c, self.coords_index[0]:self.coords_index[1]].detach().cpu().numpy()).labels_
                        for l in np.unique(labels):
                            broken_clusts.append(c[labels==l])
                    clusts = broken_clusts

        # If requested, shuffle the order in which the clusters are listed (used for debugging)
        if self.shuffle_clusters:
            random.shuffle(clusts)

        # If requested, merge images together within the batch
        if self.merge_batch:
            cluster_data, particles, batch_list = merge_batch(cluster_data, particles, self.merge_batch_size, self.merge_batch_fluc)
            batch_counts = np.unique(batch_list, return_counts=True)[1]
            result['batch_counts'] = [batch_counts]

        # If an event is missing from the input data - e.g., deghosting
        # erased everything (extreme case but possible if very few voxels)
        # then we might be miscounting batches. Ensure that batches is the
        # same length as batch_size if specified.
        batches, bcounts = np.unique(cluster_data[:,self.batch_index].detach().cpu().numpy(), return_counts=True)
        if batch_size is not None:
            new_bcounts = np.zeros(batch_size, dtype=np.int64)
            new_bcounts[batches.astype(np.int64)] = bcounts
            bcounts = new_bcounts
            batches = np.arange(batch_size)

        # Update result with a list of clusters for each batch id
        if not len(clusts):
            return {**result,
                    'clusts':    [[np.array([]) for _ in batches]],
                    'batch_ids': [np.array([])]}

        batch_ids = get_cluster_batch(cluster_data, clusts)
        clusts_split, cbids = split_clusts(clusts, batch_ids, batches, bcounts)
        result['clusts'] = [clusts_split]
        result['batch_ids'] = [batch_ids]
        if self.edge_max_count > -1:
            _, cnts = np.unique(batch_ids, return_counts=True)
            if np.sum([c*(c-1) for c in cnts]) > 2*self.edge_max_count:
                print('The complete graph is too large, must skip batch') # TODO: use logging
                return result

        # If necessary, compute the cluster distance matrix
        dist_mat, closest_index = None, None
        if np.any(self.edge_max_dist > -1) or self.network == 'mst' or self.network == 'knn':
            dist_mat, closest_index = inter_cluster_distance(cluster_data[:,self.coords_index[0]:self.coords_index[1]].float(), clusts, batch_ids, self.edge_dist_metric, self.edge_dist_algorithm, return_index=True)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0), dtype=np.int64)
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids)
        elif self.network == 'delaunay':
            import numba as nb
            edge_index = delaunay_graph(cluster_data.cpu().numpy(), nb.typed.List(clusts), batch_ids, self.batch_index, self.coords_index)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat)
        elif self.network == 'knn':
            edge_index = knn_graph(batch_ids, self.edge_knn_k, dist_mat)
        elif self.network == 'bipartite':
            clust_ids = get_cluster_label(cluster_data, clusts, self.source_col)
            group_ids = get_cluster_label(cluster_data, clusts, self.target_col)
            edge_index = bipartite_graph(batch_ids, clust_ids==group_ids, dist_mat)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # If groups is sepecified, only keep edges that belong to the same group (cluster graph)
        if groups is not None:
            mask = groups[edge_index[0]] == groups[edge_index[1]]
            edge_index = edge_index[:,mask]

        # Restrict the input graph based on edge distance, if requested
        if np.any(self.edge_max_dist > -1):
            if self.edge_max_dist.shape[0] == 1:
                edge_index = restrict_graph(edge_index, dist_mat, self.edge_max_dist)
            else:
                # Here get_cluster_primary_label is used to ensure that Michel/Delta showers are given the appropriate semantic label
                if self.source_col == 5: classes = extra_feats[:,-1].cpu().numpy().astype(int) if extra_feats is not None else get_cluster_label(cluster_data, clusts, -1).astype(int)
                if self.source_col == 6: classes = extra_feats[:,-1].cpu().numpy().astype(int) if extra_feats is not None else get_cluster_primary_label(cluster_data, clusts, -1).astype(int)
                edge_index = restrict_graph(edge_index, dist_mat, self.edge_max_dist, classes)

            # Get index of closest pair of voxels for each pair of clusters
            closest_index = closest_index[edge_index[0], edge_index[1]]

        # Update result with a list of edges for each batch id
        edge_index_split, ebids = split_edge_index(edge_index, batch_ids, batches)
        result['edge_index'] = [edge_index_split]
        if edge_index.shape[1] > self.edge_max_count:
            return result

        # Obtain node and edge features
        x = self.node_encoder(cluster_data, clusts)
        e = self.edge_encoder(cluster_data, clusts, edge_index, closest_index=closest_index)

        # If extra features are provided separately, add them
        if extra_feats is not None:
            x = torch.cat([x, extra_feats.float()], dim=1)

        # Add end points and/or local directions to node features, if requested
        if self.add_points or points is not None:
            if points is None:
                points = get_cluster_points_label(cluster_data, particles, clusts)
            x = torch.cat([x, points.float()], dim=1)
            result['start_points'] = [np.hstack([batch_ids[:,None], points[:,:3].detach().cpu().numpy()])]
            result['end_points'] = [np.hstack([batch_ids[:,None], points[:,3:].detach().cpu().numpy()])]
            if self.add_local_dirs:
                dirs_start = get_cluster_directions(cluster_data[:, self.coords_index[0]:self.coords_index[1]], points[:,:3], clusts, self.dir_max_dist, self.opt_dir_max_dist)
                if self.add_local_dirs != 'start':
                    dirs_end = get_cluster_directions(cluster_data[:, self.coords_index[0]:self.coords_index[1]], points[:,3:6], clusts, self.dir_max_dist, self.opt_dir_max_dist)
                    x = torch.cat([x, dirs_start.float(), dirs_end.float()], dim=1)
                else:
                    x = torch.cat([x, dirs_start.float()], dim=1)
            if self.add_local_dedxs:
                dedxs_start = get_cluster_dedxs(cluster_data[:, self.coords_index[0]:self.coords_index[1]], cluster_data[:,4], points[:,:3], clusts, self.dedx_max_dist)
                if self.add_local_dedxs != 'start':
                    dedxs_end = get_cluster_dedxs(cluster_data[:, self.coords_index[0]:self.coords_index[1]], cluster_data[:,4], points[:,3:6], clusts, self.dedx_max_dist)
                    x = torch.cat([x, dedxs_start.reshape(-1,1).float(), dedxs_end.reshape(-1,1).float()], dim=1)
                else:
                    x = torch.cat([x, dedxs_start.reshape(-1,1).float()], dim=1)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=cluster_data.device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=cluster_data.device)

        result['node_features'] = [[x[b] for b in cbids]]
        result['edge_features'] = [[e[b] for b in ebids]]

        # Pass through the model, update results
        out = self.gnn_model(x, index, e, xbatch)
        
        # Unwrapped per image
        result['node_pred'] = [[out['node_pred'][0][b] for b in cbids]]
        result['edge_pred'] = [[out['edge_pred'][0][b] for b in ebids]]
        
        print(out['node_pred'][0].shape)
        print(out['edge_pred'][0].shape)
        
        print(len(result['node_pred'][0]))
        print(len(result['edge_pred'][0]))

        return result
    
    
class ParticleNetLoss(torch.nn.modules.loss._Loss):
    
    RETURNS = {
        'loss': ['scalar'],
        'node_loss': ['scalar'],
        'edge_loss': ['scalar'],
        'accuracy': ['scalar'],
        'node_accuracy': ['scalar'],
        'edge_accuracy': ['scalar']
    }
    

    def __init__(self, cfg, name='particlenet_loss'):
        super(ParticleNetLoss, self).__init__()

        self.batch_index = BATCH_COL
        self.coords_index = COORD_COLS
        
        self.edge_loss_weight = cfg[name].get('edge_loss_weight', 0.0)

        # Initialize the node and edge losses, if requested
        self.apply_node_loss, self.apply_edge_loss = False, False
        if 'node_loss' in cfg[name]:
            self.apply_node_loss = True
            self.node_loss = node_loss_construct(cfg[name], batch_col=BATCH_COL, coords_col=COORD_COLS)
            self.RETURNS.update(self.node_loss.RETURNS)
        if 'edge_loss' in cfg[name]:
            self.apply_edge_loss = True
            self.edge_loss = edge_loss_construct(cfg[name], batch_col=BATCH_COL, coords_col=COORD_COLS)
            self.RETURNS.update(self.edge_loss.RETURNS)
            
    def forward(self, result, clust_label, graph=None, node_label=None, iteration=None):

        # Apply edge and node losses, if instantiated
        loss = {}
        if self.apply_node_loss:
            if node_label is None:
                node_label = clust_label
            if iteration is not None:
                node_loss = self.node_loss(result, node_label, iteration=iteration)
            else:
                node_loss = self.node_loss(result, node_label)
            loss.update(node_loss)
            loss['node_loss'] = node_loss['loss']
            loss['node_accuracy'] = node_loss['accuracy']
        if self.apply_edge_loss:
            edge_loss = self.edge_loss(result, clust_label, graph)
            loss.update(edge_loss)
            loss['edge_loss'] = edge_loss['loss']
            loss['edge_accuracy'] = edge_loss['accuracy']
        if self.apply_node_loss and self.apply_edge_loss:
            loss['loss'] = loss['node_loss'] + loss['edge_loss']
            loss['accuracy'] = (loss['node_accuracy'] + loss['edge_accuracy'])/2

        return loss