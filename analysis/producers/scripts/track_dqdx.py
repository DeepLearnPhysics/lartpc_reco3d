from collections import OrderedDict

from scipy.spatial.distance import cdist

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

from mlreco.utils.gnn.cluster import cluster_dedx
from mlreco.utils.tracking import get_track_segment_dedxs
from mlreco.utils.globals import MICHL_SHP

import numpy as np

@write_to(['track_true_rr', 'track_reco_rr'])
def track_dqdx(data_blob, res, **kwargs):
    """
    """

    particles_true, particles_pred = [], []

    image_idxs = data_blob['index']
    
    step_size = kwargs['step_size']
    michel_threshold = kwargs['michel_threshold']

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            'file_index': data_blob['file_index'][idx]
        }
        
        pmatches, pcounts = res['matched_particles_t2r'][idx], res['particle_match_overlap_t2r'][idx]
        truth_particles = res['truth_particles'][idx]
        for i, pair in enumerate(pmatches):
            true_p, pred_p = pair[0], pair[1]
            
            if pred_p is None:
                continue
            
            if len(true_p.points) < 10 or len(pred_p.points) < 10:
                continue
            
            # Select Stopping muons
            children_counts = sum(true_p.children_counts[0:2])
            
            if true_p.pid == 2 and true_p.is_contained:
                # Check for presence of Michel electron
                attached_to_Michel = False
                closest_point = None
                for p in truth_particles:
                    if p.semantic_type != MICHL_SHP: continue
                    d = cdist(pred_p.points, p.points)
                    if d.min() >= michel_threshold: continue
                    attached_to_Michel = True
                    closest_point = d.min(axis=1).argmin()

                if not attached_to_Michel: continue
                
                if pred_p.pid == 2 and pred_p.is_contained:
                        
                    true_ADC, true_ADC_err, true_rrange, _, _, true_l = get_track_segment_dedxs(
                        true_p.points, true_p.depositions, true_p.end_point, step_size)
                    pred_ADC, pred_ADC_err, pred_rrange, _, _, pred_l = get_track_segment_dedxs(
                        pred_p.points, pred_p.depositions, pred_p.end_point, step_size)
                    
                    print(true_ADC)
                        
                    for i in range(len(true_ADC)):
                        update_dict = index_dict.copy()
                        out = {
                            'true_particle_id': true_p.id,
                            'true_ADC': true_ADC[i],
                            'true_ADC_err': true_ADC_err[i],
                            'true_rrange': true_rrange[i],
                            'true_l': true_l[i],
                            'true_children_counts': children_counts
                        }
                        update_dict.update(out)
                        particles_true.append(update_dict)
                        
                    for i in range(len(pred_ADC)):
                        update_dict = index_dict.copy()
                        out = {
                            'pred_particle_id': pred_p.id,
                            'pred_ADC': pred_ADC[i],
                            'pred_ADC_err': pred_ADC_err[i],
                            'pred_rrange': pred_rrange[i],
                            'pred_l': pred_l[i]
                        }
                        update_dict.update(out)
                        particles_pred.append(update_dict)
    return [particles_true, particles_pred]

@write_to(['track_reco_rr'])
def track_dqdx_data(data_blob, res, **kwargs):
    """
    """

    particles = []
    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]
    step_size = kwargs['step_size']
    michel_threshold = kwargs['michel_threshold']

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            'file_index': data_blob['file_index'][idx]
        }
        
        reco_particles = res['particles'][idx]
        for pred_p in reco_particles:
            
            if pred_p.pid == 2 and pred_p.is_contained and pred_p.is_primary:
                
                # Check for presence of Michel electron
                attached_to_Michel = False
                closest_point = None
                for p in reco_particles:
                    if p.semantic_type != MICHL_SHP: continue
                    d = cdist(pred_p.points, p.points)
                    if d.min() >= michel_threshold: continue
                    attached_to_Michel = True
                    closest_point = d.min(axis=1).argmin()

                if not attached_to_Michel: continue
                
                pred_ADC, pred_ADC_err, pred_rrange, _, _, pred_l = get_track_segment_dedxs(
                    pred_p.points, pred_p.depositions, pred_p.end_point, step_size)
                    
                for i in range(len(pred_ADC)):
                    update_dict = index_dict.copy()
                    out = {
                        'pred_particle_id': pred_p.id,
                        'pred_ADC': pred_ADC[i],
                        'pred_ADC_err': pred_ADC_err[i],
                        'pred_rrange': pred_rrange[i],
                        'pred_l': pred_l[i]
                    }
                    update_dict.update(out)
                    particles.append(update_dict)
                    
    return [particles]