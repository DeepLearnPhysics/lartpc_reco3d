import numpy as np
import numba as nb
from typing import Tuple

@nb.njit
def select_valid_domains(cumprod : nb.boolean[:, :, :]) -> Tuple[nb.boolean[:, :], nb.int64[:]]:
    
    num_particles   = cumprod.shape[0]
    num_constraints = cumprod.shape[1]
    domain_size     = cumprod.shape[2]
    
    out = np.ones((num_particles, domain_size), dtype=nb.boolean)
    index = np.zeros(num_particles, dtype=nb.int64)
    
    J = num_constraints-1
    
    for i in range(num_particles):
        found = False
        if cumprod[i, J, :].sum() >= 1:
            found = True
            out[i, :] = cumprod[i, J, :]
            index[i] = J
        else:
            num_satisfied = 0
            # We try to find the first row that has at least one solution.
            for j in range(num_constraints):
                if cumprod[i, j, :].sum() == 0:
                    # No solution at jth constraint
                    break
                else:
                    # We have a solution at jth constraint
                    num_satisfied = j
                    continue
            out[i, :] = cumprod[i, num_satisfied, :]
            index[i] = num_satisfied
    
    return out, index+1