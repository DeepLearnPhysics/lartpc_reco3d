import numpy as np
from collections import defaultdict, namedtuple
from .utils import select_valid_domains

Assignment = namedtuple('Assignment', ['var_name', 'id', 'consistency', 'score', 'value'])

class CSAT:
    """Constraint Satisfaction Interface

    Raises
    ------
    ValueError
        _description_
    """
    
    # List of particle variable names
    _PVAR_NAMES = {'pid_scores' : 'pid', 'primary_scores' : 'is_primary'}
    
    def __init__(self, interaction):
        """Initializes the CSAT object

        Parameters
        ----------
        interaction : Interaction
            Interaction object to run constraint satisfaction.
        """
        self.interaction   = interaction
        self.particles     = sorted(list(interaction.particles), 
                                    key=lambda x: x.id)
        self.particle_ids  = [p.id for p in self.particles]
        
        self.pid_to_index  = {p.id : i for i, p in enumerate(self.particles)}
        self.index_to_pid  = {i : p.id for i, p in enumerate(self.particles)}
        
        self.num_particles = len(self.particles)
        
        self._constraints  = {}
        self._solutions  = {}
        
        self._consistencies      = {}
        self._satisfied          = {}
        self._num_satisfied      = {}
        self._scores             = {}
        self._domains            = {}
        
        
    def define_variable(self, var_name, domain):
        """Defines a particle variable for constraint satisfaction

        Parameters
        ----------
        var_name : str
            Name of the particle variable to be defined.
        domain : list
            List of possible values for the particle variable.
        """
        if var_name not in self._PVAR_NAMES:
            raise ValueError(f"Variable name {var_name} not in list of particle variable names")
        # Domain of a variable is a D x 1 array (possible values)
        self._domains[var_name] = np.array(domain, dtype=int)
        # Scores of a variable is a N x D array (particle x possible scores)
        self._scores[var_name] = np.zeros((self.num_particles, len(domain)))
        self._constraints[var_name] = []
        
        for i, p in enumerate(self.particles):
            scores = getattr(p, var_name)
            assert len(scores) == len(domain)
            self._scores[var_name][i] = scores
                
                
    def define_particle_constraint(self, cst, var_name):
        """Adds a constraint to the CSAT object

        Parameters
        ----------
        cst : Constraint
            Constraint to be added to the CSAT object.
        """
        if var_name is not None:
            assert var_name in self._PVAR_NAMES
            assert var_name == cst.var_name
        self._constraints[cst.var_name].append(cst)
        
    def define_global_constraint(self, cst, var_name):
        """Adds a global constraint to the CSAT object

        Parameters
        ----------
        cst : Constraint
            Constraint to be added to the CSAT object.
        """
        self._constraints[var_name].append(cst)
        
    def _process_constraints(self):
        """Processes the constraints and updates the consistency scores.
        
        Each constraint outputs a filter for each particle, which is then
        summed to get the consistency score for each particle.
        """
        
        self._constraints = {var_name: sorted(self._constraints[var_name], 
                                   key=lambda x: x.priority, 
                                   reverse=True) for var_name in self._constraints}
        self._num_constraints = { var_name : max(1, len(self._constraints[var_name])) for var_name in self._constraints}

        # Build Buffers
        for var_name, _ in self._constraints.items():
            # N x C x D array (particles x constraints x domain)
            self._consistencies[var_name] = np.ones((self.num_particles, 
                                                     self._num_constraints[var_name], 
                                                     len(self._domains[var_name]))).astype(bool)
        
        for var_name, constraints in self._constraints.items():
            for cst_id, cst in enumerate(constraints):
                if cst.scope == 'particle':
                    self._process_particle_constraints(cst, cst_id, var_name)
                elif cst.scope == 'global':
                    self._process_global_constraints(cst, cst_id, var_name)
                else:
                    raise ValueError(f"Unknown scope {cst.scope} for constraint {cst}")
                

    def _process_particle_constraints(self, cst, cst_id, var_name):
        for i, p in enumerate(self.particles):
            filters = cst(p, self.interaction)
            self._consistencies[var_name][i, cst_id, :] = filters.astype(bool)
            
    def _process_global_constraints(self, cst, cst_id, var_name):
        
        args = [getattr(self, arg) for arg in cst._DATA_CAPTURE]
        filter_map = cst(*args)
        self._consistencies[var_name][:, cst_id, :] = filter_map
            
                
    def solve(self, min_consistency=None, debug=False):
        
        self._process_constraints()
        
        for var_name, cmap in self._consistencies.items():
            solution_map = np.cumprod(cmap, axis=1)
            out, index = select_valid_domains(solution_map)
            if len(self._constraints[var_name]) == 0:
                index -= 1
            self._num_satisfied[var_name] = index
            
            values = self._scores[var_name] * out.astype(int)
            self._solutions[var_name] = values.max(axis=1)
            self._solutions[self._PVAR_NAMES[var_name]] = values.argmax(axis=1)
            
        for var_name, nsat in self._num_satisfied.items():
            if var_name not in self._satisfied:
                self._satisfied[var_name] = defaultdict(list)
            for i, part_id in self.index_to_pid.items():
                for j in range(nsat[i]):
                    cst_name = self._constraints[var_name][j].__repr__()
                    self._satisfied[var_name][part_id].append(cst_name)
                
            
    def get_assignment(self, var_name, pid):
        i = self.pid_to_index[pid]
        return self._solutions[var_name][i]
    
    def get_satisfiability(self, var_name=None):
        if var_name not in self._PVAR_NAMES:
            raise ValueError(f"Variable name {var_name} not in list of particle variable names")
        N = self._num_satisfied[var_name] / self._num_constraints[var_name]
        score = N * self.solutions[var_name]
        return score.mean()
    
    @property
    def satisfiability(self):
        out = []
        for var_name in self._PVAR_NAMES:
            out.append(self.get_satisfiability(var_name))
        return sum(out) / len(out)
            
        
    def __repr__(self):
        msg = '''CSAT(Constraints: {})'''.format(str(self._constraints))
        return msg
    
    @property
    def constraints(self):
        return self._constraints
    
    @property
    def solutions(self):
        return self._solutions
    
    @property
    def consistencies(self):
        return self._consistencies
    
    @property
    def scores(self):
        return self._scores
    
    @property
    def domains(self):
        return self._domains
    
    @property
    def num_satisfied(self):
        return self._num_satisfied
    
    @property
    def num_constraints(self):
        return self._num_constraints