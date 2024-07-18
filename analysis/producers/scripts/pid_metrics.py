from analysis.producers.decorator import write_to
import numpy as np
from mlreco.utils.globals import *

from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.tracking import get_track_segment_dedxs, get_track_deposition_gradient
from mlreco.utils.gnn.cluster import get_cluster_dedxs, get_cluster_points_label

@write_to(['pid_metrics'])
def pid_metrics(data_blob, res, **kwargs):
    """
    Select particle pairs for logging (only for mpv/nu interactions)
    """

    metrics = []
    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):
        
        if 'node_pred' in res:
            logits = res['node_pred'][idx]
        elif 'logits' in res:
            logits = res['logits'][idx]
        clusts = res['clusts'][idx]
        
        particles = { p.id(): p for p in data_blob['particles_asis'][idx] }
        
        data = data_blob['input_data'][idx]
        
        # print(data[])
        
        labels = get_cluster_label(data, clusts, column=PID_COL)
        nu_labels = get_cluster_label(data, clusts, column=NU_COL)
        pgrps = get_cluster_label(data, clusts, column=PGRP_COL)
        pshows = get_cluster_label(data, clusts, column=PSHOW_COL)
        group_labels = get_cluster_label(data, clusts, column=GROUP_COL)
        shape_labels = get_cluster_label(data, clusts, column=SHAPE_COL)
        
        partice_points = data_blob['particles_label'][idx]
        
        points = get_cluster_points_label(data, partice_points, clusts, random_order=False)
        
        shower_dedx = {}
        
        shower_dedx[0] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=3)
        shower_dedx[1] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=5)
        shower_dedx[2] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=7.5)
        shower_dedx[3] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=10)
        shower_dedx[4] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=15)
        shower_dedx[5] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=20)
        shower_dedx[6] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=30)
        shower_dedx[7] = get_cluster_dedxs(data[:, COORD_COLS], data[:, VALUE_COL], points[:, :3], clusts, max_dist=40)

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
            for dedx_length in shower_dedx:
                out_dict[f'shower_dedx_{dedx_length}'] = shower_dedx[dedx_length][j]
            out_dict['semantic_type'] = int(shape_labels[j])
            for k, s in enumerate(score):
                out_dict['Score_{}'.format(k)] = s
            metrics.append(out_dict)
    return [metrics]


@write_to(['singlep_metrics'])
def singlep_metrics(data_blob, res, **kwargs):
    """
    Select particle pairs for logging (only for mpv/nu interactions)
    """

    metrics = []
    image_idxs = data_blob['index']
    meta = data_blob['meta'][0]
    
    particles = data_blob['particles_asis'][0]
    
    particle_info = {
        'creation_process': None,
        'energy_init': None,
        'energy_deposit': None
    }
    
    out = {
        'Index': image_idxs[0],
        'Truth': int(data_blob['label'][0][:, 1]),
        'Prediction': np.argmax(res['logits'][0])
    }
    
    points = data_blob['input_data'][0][:, COORD_COLS]
    values = data_blob['input_data'][0][:, VALUE_COL]
    semantic_type = data_blob['input_data'][0][:, SHAPE_COL]
    
    
    for p in particles:
        if p.id() == p.parent_id():
            particle_info['creation_process'] = p.creation_process()
            particle_info['energy_init'] = p.energy_init()
            particle_info['energy_deposit'] = p.energy_deposit()
            particle_info['ke'] = p.energy_init() - PID_MASSES[out['Truth']]
            
            dist, dedx = compute_min_dist_to_startpoint(points, values, p, meta, r=5.0)
            particle_info['min_dist'] = dist
            particle_info['dedx'] = dedx
            particle_info['pdg_code'] = p.pdg_code()
            
        
    out.update(particle_info)
    
    metrics.append(out)
    return [metrics]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def pixel_to_cm_1d(vec, meta):
    out = np.zeros_like(vec)
    out[0] = meta[0] + meta[6] * vec[0]
    out[1] = meta[1] + meta[7] * vec[1]
    out[2] = meta[2] + meta[8] * vec[2]
    return out

def compute_min_dist_to_startpoint(points, values, particle_asis, meta, r=5.0):
    
    startpoint = np.array([getattr(particle_asis.first_step(), x)() for x in ['x', 'y', 'z']] )
    
    startpoint = pixel_to_cm_1d(startpoint, meta)
    
    dists = np.linalg.norm(points - startpoint, axis=1)
    min_dist = dists.min()
    min_index = np.argmin(dists)
    
    closest_point = points[min_index]
    new_dists = np.linalg.norm(points - closest_point, axis=1)
    
    dedx = (values[new_dists < r].sum() + 1e-6) / (new_dists[new_dists < r].max() + 1e-6)
    
    return min_dist, dedx
    