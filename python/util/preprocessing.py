import numpy as np 

def sample_down(ts_input,size=0.05):
    ts = ts_input.copy()
    factor = len(ts) // int(size*len(ts))
    sample = sample[:len(sample) // factor * factor].reshape(-1, factor).mean(axis=1)
    return sample 


def zNormalize(ts):
    ts -= np.mean(ts)
    std = np.std(ts)
    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std
    return ts