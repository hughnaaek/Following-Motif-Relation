__all__ = ['sample_down', 'zNormalize']

import numpy as np 

def sample_down(ts_input,size=0.05):
    ts = ts_input.copy()
    factor = len(ts) // int(size*len(ts))
    sample = ts[:len(ts) // factor * factor].reshape(-1, factor).mean(axis=1)
    return sample 

def zNormalize(ts):
    ts -= np.mean(ts)
    std = np.std(ts)
    
    return ts/std
