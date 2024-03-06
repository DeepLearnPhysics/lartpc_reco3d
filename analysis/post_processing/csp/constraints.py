import numpy as np
from abc import abstractmethod
from mlreco.utils.globals import *

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
        
        out = np.ones(self.domain_size).astype(int)
        if particle.semantic_type != 0:
            return out
        dists = np.linalg.norm(particle.points - interaction.vertex, axis=1)
        # Check if particle point cloud is separated from vertex:
        if dists.any() >= self.r:
            out[ELEC_PID] = 0

        return out
    
    def __repr__(self):
        return (
            'EMVertexConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}, '
            'r={}'
            ')'.format(self.scope, self.var_name, self.domain_size, self.r)
        )
    
    
class ProtonScoreConstraint(ParticleConstraint):
    
    def __init__(self, scope='particle', var_name='pid_scores', 
                 domain_size=5, threshold=0.1, priority=0):
        super(ProtonScoreConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        self.threshold = threshold
        
    def __call__(self, particle, interaction=None):
        out = np.ones(self.domain_size).astype(int)
        if particle.pid_scores[PROT_PID] < self.threshold:
            out[PROT_PID] = 0
        return out
    
    def __repr__(self):
        return (
            'ProtonScoreConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}, '
            'threshold={}'
            ')'.format(self.scope, self.var_name, self.domain_size, self.threshold)
        )
    
class PIDScoreConstraint(ParticleConstraint):
    def __init__(self, scope='particle', var_name='pid_scores', 
                 domain_size=5, priority=0):
        super(PIDScoreConstraint, self).__init__(scope, priority)
        self.domain_size = domain_size
        self.var_name = var_name
        
    def __call__(self, particle, interaction=None):
        return (particle.pid_scores > 0).astype(int)
        
    def __repr__(self):
        return (
            'PIDHardConstraint('
            'scope={}, '
            'var_name={}, '
            'domain_size={}'
            ')'.format(self.scope, self.var_name, self.domain_size)
        )
    
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
    
    def __init__(self, priority=0):
        self.priority = priority
        if priority >= 0:
            msg = "Global constraints must have negative priority. "\
                "This is to ensure that they are processed last."
            raise ValueError(msg)
        
    @abstractmethod
    def __call__(self, solver):
        raise NotImplementedError
        

class MuonElectronConstraint(GlobalConstraint):
    
    _DATA_CAPTURE = ['allowed', 'scores', 'assignments']
    
    def __init__(self, solver, priority=-1):
        super(MuonElectronConstraint, self).__init__(priority)
        if priority >= 0:
            msg = "Global constraints must have negative priority. "\
                "This is to ensure that they are processed last."
            raise ValueError(msg)
        for key in self._DATA_CAPTURE:
            setattr(self, key, getattr(solver, key))
        
    def __call__(self):
        primary_mask = self.assignments['primary_scores'].astype(bool)
        