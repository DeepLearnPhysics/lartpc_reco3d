import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Batch, Data

from mlreco.models.layers.common.cnn_encoder import SparseResidualEncoder
from mlreco.models.experimental.layers.pointnet import PointNetEncoder

from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.experimental.bayes.encoder import MCDropoutEncoder
from mlreco.utils.gnn.data import split_clusts
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_label

from mlreco.models.experimental.losses.classification import *
from mlreco.utils.globals import PID_COL

class ParticleImageClassifier(nn.Module):

    MODULES = ['particle_image_classifier', 'network_base', 'mink_encoder']

    def __init__(self, cfg, name='particle_image_classifier'):
        super(ParticleImageClassifier, self).__init__()

        # Get the config
        model_cfg = cfg.get(name, {})

        # Initialize encoder
        self.encoder_type = model_cfg.get('encoder_type', 'standard')
        if self.encoder_type == 'dropout':
            self.encoder = MCDropoutEncoder(cfg)
        elif self.encoder_type == 'standard':
            self.encoder = SparseResidualEncoder(cfg)
        elif self.encoder_type == 'pointnet':
            self.encoder = PointNetEncoder(cfg)
        else:
            raise ValueError('Unrecognized encoder type: {}'.format(self.encoder_type))

        # Initialize final layer
        self.num_classes = model_cfg.get('num_classes', 5)
        self.final_layer = nn.Linear(self.encoder.latent_size, self.num_classes)
        print('Total Number of Trainable Parameters = {}'.format(
                    sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        point_cloud, = input
        out = self.encoder(point_cloud)
        out = self.final_layer(out)
        res = {
            'logits': [out]
        }
        return res


class MultiParticleImageClassifier(ParticleImageClassifier):

    MODULES = ['particle_image_classifier', 'network_base', 'mink_encoder']

    def __init__(self, cfg, name='particle_image_classifier'):
        super(MultiParticleImageClassifier, self).__init__(cfg, name)

        model_cfg = cfg.get(name, {})
        self.batch_col = model_cfg.get('batch_col', 0)
        self.split_col = model_cfg.get('split_col', 6)
        self.num_classes = model_cfg.get('num_classes', 5)

        self.skip_invalid = model_cfg.get('skip_invalid', True)
        self.target_col   = model_cfg.get('target_col', 9)
        self.invalid_id   = model_cfg.get('invalid_id', -1)

        self.split_input_mode = model_cfg.get('split_input_as_tg_batch', False)

    def split_input_as_tg_batch(self, point_cloud, clusts=None):
        point_cloud_cpu  = point_cloud.detach().cpu().numpy()
        batches, bcounts = np.unique(point_cloud_cpu[:,self.batch_col], return_counts=True)
        if clusts is None:
            clusts = form_clusters(point_cloud_cpu, column=self.split_col)
        if not len(clusts):
            return Batch()
        
        if self.skip_invalid:
            target_ids = get_cluster_label(point_cloud_cpu, clusts, column=self.target_col)
            clusts = [c for i, c in enumerate(clusts) if target_ids[i] != self.invalid_id]
            if not len(clusts):
                return Batch()
        
        data_list = []
        for i, c in enumerate(clusts):
            x = point_cloud[c, 4].view(-1, 1)
            pos = point_cloud[c, 1:4]
            data = Data(x=x, pos=pos)
            data_list.append(data)
        
        split_data = Batch.from_data_list(data_list)
        return split_data, clusts   


    def split_input(self, point_cloud, clusts=None):
        point_cloud_cpu  = point_cloud.detach().cpu().numpy()
        batches, bcounts = np.unique(point_cloud_cpu[:,self.batch_col], return_counts=True)
        if clusts is None:
            clusts = form_clusters(point_cloud_cpu, column=self.split_col)
        if not len(clusts):
            return point_cloud, [np.array([]) for _ in batches], []

        if self.skip_invalid:
            target_ids = get_cluster_label(point_cloud_cpu, clusts, column=self.target_col)
            clusts = [c for i, c in enumerate(clusts) if target_ids[i] != self.invalid_id]
            if not len(clusts):
                return point_cloud, [np.array([]) for _ in batches], []

        split_point_cloud = point_cloud.clone()
        split_point_cloud[:, self.batch_col] = -1
        for i, c in enumerate(clusts):
            split_point_cloud[c, self.batch_col] = i
        
        batch_ids = get_cluster_label(point_cloud_cpu, clusts, column=self.batch_col)
        clusts_split, cbids = split_clusts(clusts, batch_ids, batches, bcounts)

        return split_point_cloud[split_point_cloud[:,self.batch_col] > -1], clusts_split, cbids


    def forward(self, input, clusts=None):
        res = {}
        point_cloud, = input
        
        # It is possible that pid = 5 appears in the 9th column.
        # In that case, it is observed that the training crashses with a
        # integer overflow numel error. 
        mask = point_cloud[:, PID_COL] < self.num_classes
        valid_points = point_cloud[mask]
        
        if self.split_input_mode:
            batch, clusts = self.split_input_as_tg_batch(valid_points, clusts)
            out = self.encoder(batch)
            out = self.final_layer(out)
            res['clusts'] = [clusts]
            res['logits'] = [out]
        else:
            out, clusts_split, cbids = self.split_input(valid_points, clusts)
            res['clusts'] = [clusts_split]

            out = self.encoder(out)
            out = self.final_layer(out)
            res['logits'] = [[out[b] for b in cbids]]

        return res


def classification_loss_dict(name):
    """Loss function dictionary for classification tasks.

    All loss functions assume that the inputs are of the form
    (logits, labels), where logits is a tensor of shape (N, C) and
    labels is a tensor of shape (N,) containing the class labels.
    
    Also assumes that reduction is set to 'mean' in the loss function.
    """
    
    loss_dict = {
        'cross_entropy': CrossEntropyLoss,
        'focal': FocalLoss,
        'dice': DiceLoss,
        'lovasz_softmax': LovaszSoftmaxLoss,
        'jaccard': JaccardLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'log_cosh_dice': LogCoshDiceLoss,
        'log_dice': LogDiceLoss
    }
    
    constructor = loss_dict[name]
    
    print(f"Initialized {constructor.__name__} Loss for Classification.")
    
    return constructor


class ParticleTypeLoss(nn.Module):

    def __init__(self, cfg, name='particle_type_loss'):
        super(ParticleTypeLoss, self).__init__()
        self.loss_fn_name = cfg.get(name, {}).get('loss', 'cross_entropy')
        loss_kwargs = cfg.get(name, {}).get('loss_kwargs', {})
        self.loss_fn = classification_loss_dict(self.loss_fn_name)(**loss_kwargs)
        self.num_classes = cfg.get(name, {}).get('num_classes', 5)

    def forward(self, out, type_labels):
        # print(type_labels)
        logits = out['logits'][0]
        labels = type_labels[0][:, -1].to(dtype=torch.long)

        loss = self.loss_fn(logits, labels)

        pred   = torch.argmax(logits, dim=1)
        accuracy = float(torch.sum(pred[labels > -1] == labels[labels > -1])) / float(labels[labels > -1].shape[0])

        res = {
            'loss': loss,
            'accuracy': accuracy
        }

        for c in range(self.num_classes):
            mask = labels == c
            res[f'accuracy_class_{c}'] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask)) \
                if torch.sum(mask) else 1.

        return res


class MultiParticleTypeLoss(nn.Module):

    def __init__(self, cfg, name='particle_type_loss'):
        super(MultiParticleTypeLoss, self).__init__()

        loss_cfg = cfg.get(name, {})
        self.num_classes = loss_cfg.get('num_classes', 5)
        self.batch_col   = loss_cfg.get('batch_col',   0)
        self.target_col  = loss_cfg.get('target_col', 9)
        self.balance_classes = loss_cfg.get('balance_classes', False)
        self.loss_fn_name = loss_cfg.get('loss', 'cross_entropy')
        
        loss_kwargs = loss_cfg.get('loss_kwargs', {})
        
        if self.loss_fn_name != 'cross_entropy' and self.balance_classes:
            raise ValueError('Only cross entropy loss can be balanced.')

        if self.balance_classes:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        else:
            self.loss_fn = classification_loss_dict(self.loss_fn_name)(**loss_kwargs)

        self.split_input_mode = loss_cfg.get('split_input_as_tg_batch', False)

    def forward_tg(self, out, valid_labels):

        logits = out['logits'][0]
        clusts = out['clusts'][0]

        labels = get_cluster_label(valid_labels, clusts, self.target_col)

        return [logits], [labels]


    def forward(self, out, type_labels):
    
        valid_labels = type_labels[0][type_labels[0][:, 9] < self.num_classes]

        if self.split_input_mode:
            logits, labels = self.forward_tg(out, valid_labels)

        else:
            logits = out['logits'][0]
            clusts = out['clusts'][0]
            labels = [get_cluster_label(valid_labels[valid_labels[:, self.batch_col] == b], 
                        clusts[b], self.target_col) for b in range(len(clusts)) if len(clusts[b])]

        if not len(labels):
            res = {
                'loss': torch.tensor(0., requires_grad=True, device=valid_labels.device),
                'accuracy': 1.,
                'reg_loss': 0.0,
                'ce_loss': 0.0
            }
            for c in range(self.num_classes):
                res[f'accuracy_class_{c}'] = 1.
            return res

        labels = torch.tensor(np.concatenate(labels), dtype=torch.long, device=valid_labels.device)
        logits = torch.cat(logits, axis=0)
        
        res = {
            'loss': torch.tensor(0., requires_grad=True, device=valid_labels.device),
            'accuracy': 1.,
            'reg_loss': 0.0,
            'ce_loss': 0.0
        }

        if not self.balance_classes:
            output = self.loss_fn(logits, labels)
            loss = output['loss']
            res['reg_loss'] = output['reg_loss']
            res['ce_loss'] = output['ce_loss']
        else:
            classes, counts = labels[labels>-1].unique(return_counts = True)
            weights = torch.sum(counts)/counts/self.num_classes
            loss = 0.
            for i, c in enumerate(classes):
                class_mask = labels == c
                loss += weights[i] * self.loss_fn(logits[class_mask], labels[class_mask]) / torch.sum(counts)

        pred   = torch.argmax(logits, dim=1)
        accuracy = float(torch.sum(pred[labels > -1] == labels[labels > -1])) / float(labels[labels > -1].shape[0])
        
        res['loss'] = loss
        res['accuracy'] = accuracy

        for c in range(self.num_classes):
            mask = labels == c
            res[f'accuracy_class_{c}'] = \
                float(torch.sum(pred[mask] == labels[mask])) / float(torch.sum(mask)) \
                if torch.sum(mask) else 1.

        return res