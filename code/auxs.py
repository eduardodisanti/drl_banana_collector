#######################
# AUXILIARY FUNCTIONS #
#######################

import numpy as np

def moving_average(a, n=10) :
    
    resp = np.cumsum(a, dtype=float)
    resp[n:] = resp[n:] - resp[:-n]
    return resp[n - 1:] / n
    
