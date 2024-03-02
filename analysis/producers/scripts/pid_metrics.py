from analysis.producers.decorator import write_to
import numpy as np
from mlreco.utils.globals import *

from mlreco.utils.gnn.cluster import get_cluster_label

@write_to(['pid_metrics'])
def pid_metrics(data_blob, res, **kwargs):
    """
    Select particle pairs for logging (only for mpv/nu interactions)
    """

    metrics = []
    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]
    
    

    for idx, index in enumerate(image_idxs):
        
        logits = res['logits'][idx]
        clusts = res['clusts'][idx]
        particles = { p.id(): p for p in data_blob['particles_asis'][idx] }
        
        data = data_blob['input_data'][idx]
        
        labels = get_cluster_label(data, clusts, column=PID_COL)
        nu_labels = get_cluster_label(data, clusts, column=NU_COL)
        pgrps = get_cluster_label(data, clusts, column=PGRP_COL)
        pshows = get_cluster_label(data, clusts, column=PSHOW_COL)
        group_labels = get_cluster_label(data, clusts, column=GROUP_COL)

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            # 'file_index': data_blob['file_index'][idx]
        }
        
        for j, score in enumerate(logits):
            group_id = group_labels[j]
            p = particles.get(group_id, None)
            if p is None:
                continue
            pred = np.argmax(score)
            truth = int(labels[j])
            out_dict = index_dict.copy()
            out_dict['Prediction'] = pred
            out_dict['Truth'] = truth
            out_dict['nu_id'] = int(nu_labels[j])
            out_dict['pgrp_id'] = int(pgrps[j])
            out_dict['pshow_id'] = int(pshows[j])
            out_dict['energy_init'] = p.energy_init()
            out_dict['energy_deposit'] = p.energy_deposit()
            out_dict['creation_process'] = p.creation_process()
            for k, s in enumerate(score):
                out_dict['Score_{}'.format(k)] = s
            metrics.append(out_dict)
    return [metrics]