import numpy as np
from abc import abstractmethod
from mlreco.utils.globals import *
from .utils import select_valid_domains

import networkx as nx

def constraints_dict(name):
    cst_dict = {
        'particle_semantic_constraint': ParticleSemanticConstraint,
        'primary_semantic_constraint': PrimarySemanticConstraint,
        'em_vertex_constraint': EMVertexConstraint,
        'particle_score_constraint': ParticleScoreConstraint,
        # 'pid_score_constraint': PIDScoreConstraint,
        'primary_constraint': PrimaryConstraint,
        'muon_electron_constraint': MuonElectronConstraint
    }
    return cst_dict[name]

class ParticleConstraint:
    
    def __init__(self, scope, var_name=None, priority=0):
        """Constructor for the ParticleConstraint.

        Parameters
        ----------
        scope : str
            The scope of the constraint, either 'particle' or 'global'.
        var_name : str
            The name of the variable the constraint operates on.
        priority : int, optional
            The priority of the constraint. The higher the priority, 
            the earlier the constraint is processed. The default is 0.
        """
        self._scope = scope
        self.priority = priority
        
    @property
    def scope(self):
        return self._scope
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """A particle constraint takes a particle and an interaction
        and returns a filter for the particle's domain.

        Returns
        -------
        np.ndarray
            D x 0 array of booleans, where D is the domain size.

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError
    
    
class ParticleSemanticConstraint(ParticleConstraint):
    """Enforces semantic constraints on particles types. 

    """
    name = 'particle_semantic_constraint'
    
    def __init__(self, scope='particle', var_name='pid_scores', 
                 domain_size=5, priority=0):
        super(ParticleSemanticConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        
    def __call__(self, particle, *args, **kwargs):
    
        out = np.ones(self.domain_size).astype(bool)
        if particle.semantic_type == 0:
            # Showers cannot be muons, protons, or pions.
            out[MUON_PID:PROT_PID+1] = False
            out[PHOT_PID:ELEC_PID+1] = True
        elif particle.semantic_type == 1:
            # Tracks cannot be photons or electrons.
            out[MUON_PID:PROT_PID+1] = True
            out[PHOT_PID:ELEC_PID+1] = False
        elif particle.semantic_type == 2 or particle.semantic_type == 3:
            # Michels and Deltas must be electrons.
            out[ELEC_PID] = False
            out = np.invert(out) # out is only True in the ELEC_PID index.
        else:
            pass
        return out
    
    
class PrimarySemanticConstraint(ParticleConstraint):
    
    def __init__(self, scope='particle', var_name='primary_scores', 
                 domain_size=2, priority=0, threshold=0.1):
        super(PrimarySemanticConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        
    def __call__(self, particle, *args, **kwargs):
        
        out = np.ones(self.domain_size).astype(bool)
        if particle.primary_scores[1] >= 0.1:
            out[0] = False
            out[1] = True
        if particle.semantic_type == 2 or particle.semantic_type == 3:
            # Michels and Deltas cannot be primaries.
            out[1] = False
        return out
    
    def __repr__(self):
        return (
            'PrimarySemanticConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}, '
            'priority={}, '
            'threshold={}'
            ')'.format(self.scope, self.var_name, self.domain_size, 
                       self.priority, self.threshold)
        )
    
    
class EMVertexConstraint(ParticleConstraint):
    """Primary electron must touch interaction vertex.
    """
    
    def __init__(self, scope='particle', var_name='pid_scores', 
                 domain_size=5, r=2, priority=0):
        super(EMVertexConstraint, self).__init__(scope, priority)
        self.r = r
        self.domain_size = domain_size
        self.var_name = var_name
        
    def __call__(self, particle, interaction):
        
        out = np.ones(self.domain_size).astype(bool)
        if particle.semantic_type != 0:
            return out
        dists = np.linalg.norm(particle.points - interaction.vertex, axis=1)
        # Check if particle point cloud is separated from vertex:
        print(particle, dists.min(), dists.all())
        if (dists >= self.r).all():
            out[ELEC_PID] = False
            out[PHOT_PID] = True

        return out
    
    def __repr__(self):
        return (
            'PrimarySemanticConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}, '
            'priority={}, '
            'threshold={}'
            ')'.format(self.scope, self.var_name, self.domain_size, 
                       self.priority, self.threshold)
        )
    
    
class ParticleScoreConstraint(ParticleConstraint):
    
    def __init__(self, scope='particle', var_name='pid_scores', 
                 domain_size=5, 
                 proton_threshold=0.85, 
                 muon_threshold=0.1, 
                 pion_threshold=0.0, 
                 priority=0):
        super(ParticleScoreConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        self.proton_threshold = proton_threshold
        self.muon_threshold = muon_threshold
        self.pion_threshold = pion_threshold
        
    def __call__(self, particle, interaction=None):
        
        out = np.ones(self.domain_size).astype(bool)

        if particle.pid_scores[PROT_PID] >= self.proton_threshold:
            out = np.zeros(self.domain_size).astype(bool)
            out[PROT_PID] = True
            return out
        elif particle.pid_scores[MUON_PID] >= self.muon_threshold:
            out = np.zeros(self.domain_size).astype(bool)
            out[MUON_PID] = True
            return out
        elif particle.pid_scores[PION_PID] >= self.pion_threshold:
            out = np.zeros(self.domain_size).astype(bool)
            out[PION_PID] = True
        else:
            return out
        
        return out
    
    def __repr__(self):
        return (
            'ParticleScoreConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}, '
            'proton_threshold={}, '
            'muon_threshold={}, '
            'pion_threshold={}'
            ')'.format(self.scope, self.var_name, self.domain_size, 
                       self.proton_threshold, self.muon_threshold, 
                       self.pion_threshold)
        )
    
# class PIDScoreConstraint(ParticleConstraint):
#     def __init__(self, scope='particle', var_name='pid_scores', 
#                  domain_size=5, priority=0):
#         super(PIDScoreConstraint, self).__init__(scope, priority)
#         self.domain_size = domain_size
#         self.var_name = var_name
        
#     def __call__(self, particle, interaction=None):
#         return (particle.pid_scores > 0).astype(int)
        
#     def __repr__(self):
#         return (
#             'PIDHardConstraint('
#             'scope={}, '
#             'var_name={}, '
#             'domain_size={}'
#             ')'.format(self.scope, self.var_name, self.domain_size)
#         )
    
class PrimaryConstraint(ParticleConstraint):
    def __init__(self, scope='particle', var_name='primary_scores', 
                 domain_size=2, priority=0):
        super(PrimaryConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        
    def __call__(self, particle=None, interaction=None):
        return (particle.primary_scores > 0).astype(int)
    
    def __repr__(self):
        return (
            'PrimaryConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}'
            ')'.format(self.scope, self.var_name, self.domain_size)
        )
        
        
class GlobalConstraint:
    
    def __init__(self, priority=-1, var_name=None):
        self.priority = priority
        self.var_name = var_name
        self.scope = 'global'
        if priority >= 0:
            msg = "Global constraints must have negative priority. "\
                "This is to ensure that they are processed last."
            raise ValueError(msg)
        
    @abstractmethod
    def __call__(self, solver):
        raise NotImplementedError
        

class MuonElectronConstraint(GlobalConstraint):
    
    _DATA_CAPTURE = ['consistencies', 'scores']
    
    def __init__(self, priority=-1, var_name='pid_scores'):
        super(MuonElectronConstraint, self).__init__(priority, var_name)
        if priority >= 0:
            msg = "Global constraints must have negative priority. "\
                "This is to ensure that they are processed last."
            raise ValueError(msg)
        
    def __call__(self, consistencies, scores):

        # Allowed Primary solutions
        pid_consistencies = consistencies['pid_scores']
        primary_consistencies = consistencies['primary_scores']
        
        primary_scores = scores['primary_scores']
        pid_scores = scores['pid_scores']
        
        cumprod_pid = np.cumprod(pid_consistencies, axis=1)
        cumprod_primary = np.cumprod(primary_consistencies, axis=1)
        
        cns = self._mu_e_consistency_map(pid_consistencies, 
                                         primary_consistencies,
                                         cumprod_pid,
                                         cumprod_primary,
                                         primary_scores, pid_scores)
        
        return cns
    
    @staticmethod
    # @nb.njit
    def _mu_e_consistency_map(pid_consistencies, 
                              primary_consistencies,
                              cumprod_pid,
                              cumprod_primary,
                              primary_scores, 
                              pid_scores):

        N, C, S = pid_consistencies.shape

        out = np.ones((N, S)).astype(bool)

        pid_cmap = pid_consistencies
        primary_cmap = primary_consistencies

        valid_pids, _ = select_valid_domains(cumprod_pid)
        valid_primaries, _ = select_valid_domains(cumprod_primary)

        primary_mask = valid_primaries.argmax(axis=1).astype(bool)

        pid_score_map = pid_scores * primary_mask.reshape(-1, 1) * valid_pids
        pid_guess = np.argmax(pid_score_map, axis=1)

        counts = np.zeros(valid_pids.shape[1], dtype=np.int64)
        labels, c = np.unique(pid_guess, return_counts=True)
        counts[labels] = c

        if counts[ELEC_PID] >= 1 and counts[MUON_PID] >= 1:
            muon = (pid_score_map[:, MUON_PID].argmax(), MUON_PID)
            elec = (pid_score_map[:, ELEC_PID].argmax(), ELEC_PID)
            # Set all electron and muon consistencies to False
            out[primary_mask, MUON_PID] = False
            out[primary_mask, ELEC_PID] = False
            if pid_score_map[elec] > pid_score_map[muon]:
                # Pick one with highest electron score
                out[elec[0], elec[1]] = True
            else:
                out[muon[0], muon[1]] = True
        
        return out
    
    def __repr__(self):
        return 'MuonElectronConstraint()'