import numpy as np
from collections import defaultdict
from analysis.post_processing import post_processing
from mlreco.utils.globals import *
#from matcha.match_candidate import MatchCandidate
#from matcha.track import Track
#from matcha.crthit import CRTHit

@post_processing(data_capture=['meta', 'index', 'crthits'], 
                 result_capture=['interactions'])
def run_crt_tpc_matching(data_dict, result_dict, 
                         crt_tpc_manager=None):
    """
    Post processor for running CRT-TPC matching using matcha.
    
    Parameters
    ----------

    Returns
    -------
    update_dict: dict of list
        Dictionary of a list of length batch_size, where each entry in 
        the list is a mapping:
            interaction_id : (matcha.CRTHit, matcha.MatchCandidate)
        
    NOTE: This post-processor also modifies the list of Interactions
    in-place by adding the following attributes:
        interaction.crthit_matched: (bool)
            Indicator for whether the given interaction has a CRT-TPC match
        interaction.crthit_id: (list of ints)
            List of IDs for CRT hits that were matched to one or more tracks
    """
    crthits = {}
    assert len(crthit_keys) > 0
    for key in crthit_keys:
        crthits[key] = data_dict[key]
    update_dict = {}
    
    interactions = result_dict['interactions']
    entry        = data_dict['index']
    
    crt_tpc_matches = crt_tpc_manager.get_crt_tpc_matches(int(entry), 
                                                          interactions,
                                                          crthits,
                                                          restrict_interactions=[])

    assert all(isinstance(item, MatchCandidate) for item in crt_tpc_matches)
    print('Interactions:\n')
    print(interactions)

    # crt_tpc_matches is a list of matcha.MatchCandidates. Each MatchCandidate
    # contains a Track and CRTHit instance. The Track class contains the 
    # interaction_id.
    matched_interaction_ids = [int_id for int_id in crt_tpc_matches.track.interaction_id]
    
    matched_interactions = [i for i in interactions 
                            if i.id in matched_interaction_ids]

    update_dict = defaultdict(list)

    crt_tpc_dict = {}
    for interaction in enumerate(matched_interactions):
        crt_tpc_dict[matched_track.id] = (matched_crthit.id)
        interaction.crthit_matched = True
        # TODO Get CRTHit and Track (aka larcv.Particle) IDs
        #interaction.crthit_id = 
        update_dict['interactions'].append(interaction)
    update_dict['crt_tpc_matches'].append(crt_tpc_dict)
        
    return update_dict



















