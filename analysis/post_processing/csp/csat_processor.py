from analysis.post_processing import PostProcessor
from .constraints import constraints_dict
from .solver import CSAT

class CSATProcessor(PostProcessor):
    
    name = 'constraint_satisfaction'
    result_cap = ['particles', 'interactions']
    
    def __init__(self, run_mode='reco', 
                 variables=None, 
                 constraint_args=None):
        if run_mode != 'reco':
            raise ValueError('Constraint Satisfaction Solver is only meant for use in reco mode')
        super(CSATProcessor, self).__init__(run_mode=run_mode)
        
        if variables is None:
            raise ValueError('Variables must be defined for Constraint Satisfaction Solver')
        if constraint_args is None:
            raise ValueError('Constraints must be defined for Constraint Satisfaction Solver')
        
        self.run_mode = run_mode
        self.constraint_args = constraint_args
        self.variables = variables
        
        
    def process(self, data_dict, result_dict):
        
        interactions = result_dict['interactions']
        for ia in interactions:
            solver = CSAT(ia)
            
            for var_name, domain in self.variables.items():
                solver.define_variable(var_name, domain)
            
            for cst_name, cst_args in self.constraint_args.items():
                cst = constraints_dict(cst_name)(**cst_args)
                if cst.scope == 'particle':
                    solver.define_particle_constraint(cst, cst.var_name)
                elif cst.scope == 'global':
                    solver.define_global_constraint(cst, cst.var_name)
                    
            solver.solve()
            
            # Assign PIDs and Primary Labels
            for p in list(ia.particles):
                p.pid = solver.get_assignment('pid', p.id)
                p.is_primary = solver.get_assignment('is_primary', p.id)
            
            ia.satisfiability = solver.satisfiability
                
        return {}, {}