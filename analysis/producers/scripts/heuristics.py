from collections import OrderedDict

from analysis.producers.decorator import write_to
from analysis.classes.data import *
from analysis.producers.logger import ParticleLogger, InteractionLogger

from mlreco.utils.gnn.cluster import cluster_dedx
from mlreco.utils.tracking import get_track_deposition_gradient

from scipy.spatial.distance import cdist

import numpy as np


@write_to(['particle_heuristics_t2r', 'particle_heuristics_r2t'])
def compute_heuristics(data_blob, res, **kwargs):
    """
    """

    interactions_t2r, particles_t2r = [], []
    interactions_r2t, particles_r2t = [], []

    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})
    
    ADC_to_MeV            = kwargs['ADC_to_MeV']

    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            'file_index': data_blob['file_index'][idx]
        }

        # 1. Match Interactions and log interaction-level information
        imatches, icounts = res['matched_interactions_t2r'][idx], res['interaction_match_overlap_t2r'][idx]
        pmatches, pcounts = res['matched_particles_t2r'][idx], res['particle_match_overlap_t2r'][idx]
        
        true_to_pred_table = {}
        true_to_pred_overlap_table = {}
        
        for ip, pair in enumerate(pmatches):
            true_p, pred_p = pair[0], pair[1]
            if true_p is not None:
                true_to_pred_table[true_p.id] = pred_p
                true_to_pred_overlap_table[true_p.id] = pcounts[ip]
        
        # 1 a) Check outputs from interaction matching 
        if len(imatches) > 0:
            # 2. Process interaction level information
            interaction_logger = InteractionLogger(int_fieldnames, meta=meta)
            interaction_logger.prepare()

            for i, interaction_pair in enumerate(imatches):

                int_dict = OrderedDict()
                int_dict.update(index_dict)
                int_dict['interaction_match_overlap'] = icounts[i]
                
                true_int, pred_int = interaction_pair[0], interaction_pair[1]

                assert (type(true_int) is TruthInteraction) or (true_int is None)
                assert (type(pred_int) is Interaction) or (pred_int is None)
                
                particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
                particle_logger.prepare()
                if true_int is not None:
                    particles = list(true_int.particles)
                    for true_p in particles:
                        true_p_dict = particle_logger.produce(true_p, mode='true')
                        part_dict = OrderedDict()
                        part_dict.update(index_dict)
                        part_dict.update(true_p_dict)
                        
                        if len(true_p.match) > 0 and pred_int is not None:
                            pred_p = true_to_pred_table[true_p.id]
                            overlap = true_p.match_overlap[0]
                        else:
                            pred_p = None
                            overlap = -1
                        pred_p_dict = particle_logger.produce(pred_p, mode='reco')
                        
                        part_dict.update(pred_p_dict)
                        
                        part_dict['true_dist_to_closest_proton'] = -np.inf
                        part_dict['reco_dist_to_closest_proton'] = -np.inf
                        
                        if true_p is not None and len(true_p.points) > 0:
                                
                            # Compute start to vertex distance (True)
                            vertex = true_int.vertex if true_int is not None else np.full(3, -np.inf)
                            start_to_vertex = np.linalg.norm(true_p.start_point - vertex)
                            
                            if true_int is not None:
                                proton_points = []
                                for p in true_int.particles:
                                    if (p.pid == 4 or p.pid == 3) and p.is_primary:
                                        if len(p.points) > 0:
                                            proton_points.append(p.start_point)
                                if len(proton_points) > 0:
                                    proton_points = np.vstack(proton_points)
                                    start_to_closest_proton = cdist(true_p.points, proton_points).min()
                                else:
                                    start_to_closest_proton = np.inf
                            else:
                                start_to_closest_proton = np.inf
                            
                            part_dict['true_dist_to_vertex'] = start_to_vertex
                            part_dict['true_dist_to_closest_proton'] = start_to_closest_proton
                            
                        if pred_p is not None and (len(pred_p.points) > 0):
                            

                            # Compute start to vertex distance (Reco)
                            vertex = pred_int.vertex if pred_int is not None else np.full(3, -np.inf)
                            start_to_vertex = np.linalg.norm(pred_p.start_point - vertex)
                            
                            if pred_int is not None:
                                proton_points = []
                                for p in pred_int.particles:
                                    if (p.pid == 4 or p.pid == 3) and p.is_primary:
                                        if len(p.points) > 0:
                                            proton_points.append(p.start_point)
                                        
                                if len(proton_points) > 0 and len(pred_p.points) > 0:
                                    proton_points = np.vstack(proton_points)
                                    start_to_closest_proton = cdist(pred_p.points, proton_points).min()
                                else:
                                    start_to_closest_proton = np.inf
                            else:
                                start_to_closest_proton = np.inf
                            
                            part_dict['reco_dist_to_vertex'] = start_to_vertex
                            part_dict['reco_dist_to_closest_proton'] = start_to_closest_proton
                            
                        part_dict['particle_match_overlap'] = true_to_pred_overlap_table[true_p.id]
                        particles_t2r.append(part_dict)
                        
                        
        # 1. Match Interactions and log interaction-level information
        imatches, icounts = res['matched_interactions_r2t'][idx], res['interaction_match_overlap_r2t'][idx]
        pmatches, pcounts = res['matched_particles_r2t'][idx], res['particle_match_overlap_r2t'][idx]
        
        pred_to_true_table = {}
        pred_to_true_overlap_table = {}
        
        for ip, pair in enumerate(pmatches):
            pred_p, true_p = pair[0], pair[1]
            if pred_p is not None:
                pred_to_true_table[pred_p.id] = true_p
                pred_to_true_overlap_table[pred_p.id] = pcounts[ip]
        
        # 1 a) Check outputs from interaction matching 
        if len(imatches) > 0:
            # 2. Process interaction level information
            interaction_logger = InteractionLogger(int_fieldnames, meta=meta)
            interaction_logger.prepare()

            for i, interaction_pair in enumerate(imatches):

                int_dict = OrderedDict()
                int_dict.update(index_dict)
                int_dict['interaction_match_overlap'] = icounts[i]
                
                pred_int, true_int = interaction_pair[0], interaction_pair[1]

                assert (type(true_int) is TruthInteraction) or (true_int is None)
                assert (type(pred_int) is Interaction) or (pred_int is None)
                
                particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
                particle_logger.prepare()
                if pred_int is not None:
                    particles = list(pred_int.particles)
                    for pred_p in particles:
                        pred_p_dict = particle_logger.produce(pred_p, mode='reco')
                        part_dict = OrderedDict()
                        part_dict.update(index_dict)
                        part_dict.update(pred_p_dict)
                        
                        if len(pred_p.match) > 0:
                            true_p = pred_to_true_table[pred_p.id]
                            overlap = pred_p.match_overlap[0]
                        else:
                            true_p = None
                            overlap = -1
                        true_p_dict = particle_logger.produce(true_p, mode='true')
                        
                        part_dict.update(true_p_dict)
                        part_dict['true_dist_to_closest_proton'] = -np.inf
                        part_dict['reco_dist_to_closest_proton'] = -np.inf
                        
                        if true_p is not None and (len(true_p.points) > 0):
                            
                            # Compute start to vertex distance (True)
                            vertex = true_int.vertex if true_int is not None else np.full(3, -np.inf)
                            start_to_vertex = np.linalg.norm(true_p.start_point - vertex)
                            
                            if true_int is not None:
                                proton_points = []
                                for p in true_int.particles:
                                    if (p.pid == 4 or p.pid == 3) and p.is_primary:
                                        if len(p.points) > 0:
                                            proton_points.append(p.start_point)
                                        
                                if len(proton_points) > 0:
                                    proton_points = np.vstack(proton_points)
                                    start_to_closest_proton = cdist(true_p.points, proton_points).min()
                                else:
                                    start_to_closest_proton = -np.inf
                            else:
                                start_to_closest_proton = -np.inf
                            
                            part_dict['true_dist_to_vertex'] = start_to_vertex
                            part_dict['true_dist_to_closest_proton'] = start_to_closest_proton
                            
                        if pred_p is not None and (len(pred_p.points) > 0):
                            
                            # Compute start to vertex distance (Reco)
                            vertex = pred_int.vertex if pred_int is not None else np.full(3, -np.inf)
                            start_to_vertex = np.linalg.norm(pred_p.start_point - vertex)
                            
                            if pred_int is not None:
                                proton_points = []
                                for p in pred_int.particles:
                                    if (p.pid == 4 or p.pid == 3) and p.is_primary:
                                        if len(p.points) > 0:
                                            proton_points.append(p.start_point)
                                        
                                if len(proton_points) > 0:
                                    proton_points = np.vstack(proton_points)
                                    start_to_closest_proton = cdist(pred_p.points, proton_points).min()
                                else:
                                    start_to_closest_proton = -np.inf
                            else:
                                start_to_closest_proton = -np.inf
                            
                            part_dict['reco_dist_to_vertex'] = start_to_vertex
                            part_dict['reco_dist_to_closest_proton'] = start_to_closest_proton
                            
                        part_dict['particle_match_overlap'] = pred_to_true_overlap_table[pred_p.id]
                        particles_r2t.append(part_dict)

    return [particles_t2r, particles_r2t]


@write_to(['particle_heuristics'])
def compute_heuristics_data(data_blob, res, **kwargs):
    """
    """

    output = []

    particle_fieldnames   = kwargs['logger'].get('particles', {})
    int_fieldnames        = kwargs['logger'].get('interactions', {})
    
    ADC_to_MeV            = kwargs['ADC_to_MeV']

    image_idxs = data_blob['index']
    meta       = data_blob['meta'][0]

    for idx, index in enumerate(image_idxs):

        index_dict = {
            'Iteration': kwargs['iteration'],
            'Index': index,
            'file_index': data_blob['file_index'][idx]
        }
        
        particles = res['particles'][idx]
        interactions = res['interactions'][idx]
        
        particle_logger = ParticleLogger(particle_fieldnames, meta=meta)
        particle_logger.prepare()
        
        for ia in interactions:
            if ia is None:
                continue
            for p in ia.particles:
                if (p.is_primary and (p.pid == 1 or p.pid == 0)):
                    pred_p_dict = particle_logger.produce(p, mode='reco')
                    part_dict = OrderedDict()
                    part_dict.update(index_dict)
                    part_dict.update(pred_p_dict)
                    
                    if p is not None:
                        dedx_start = cluster_dedx(p.points, 
                                                    p.depositions * ADC_to_MeV, 
                                                    p.start_point, 
                                                    max_dist=3.3)
                        
                        dedx_end = cluster_dedx(p.points, 
                                                p.depositions * ADC_to_MeV, 
                                                p.end_point, 
                                                max_dist=3.3)
                        
                        part_dict['reco_shower_dedx_start'] = dedx_start
                        part_dict['reco_shower_dedx_end']   = dedx_end
                        
                        # Compute start to vertex distance (Reco)
                        vertex = ia.vertex if ia is not None else np.full(3, -np.inf)
                        start_to_vertex = np.linalg.norm(p.start_point - vertex)
                        
                        proton_points = []
                        for p in ia.particles:
                            if p.pid == 4:
                                if len(p.points) > 0:
                                    proton_points.append(p.points)
                                
                        if len(proton_points) > 0:
                            proton_points = np.vstack(proton_points)
                            start_to_closest_proton = np.linalg.norm(p.start_point - proton_points, axis=1).min()
                        else:
                            start_to_closest_proton = -np.inf
                        
                        part_dict['reco_dist_to_vertex'] = start_to_vertex
                        part_dict['reco_dist_to_closest_proton'] = start_to_closest_proton
                        part_dict['flash_time'] = float(ia.flash_time)
            
                    output.append(part_dict)
        

    return [output]