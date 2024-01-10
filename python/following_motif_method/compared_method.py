import numpy as np 
from dtw import *
from scipy.signal import correlate
from scipy.signal import correlation_lags


def FLICA(leading_signal,following_signal):
    alignment = dtw(following_signal,leading_signal,
                keep_internals=True
                #,window_type = "sakoechiba"
                ,window_args= {"window_size": np.ceil(len(leading_signal)*0.1)})
    
    dtwIndexVec = alignment.index1[1:len(leading_signal)]-alignment.index2[1:len(leading_signal)]
    follVal = np.mean(np.sign(dtwIndexVec))
    
    return follVal>0


def max_correlation(x, y):
    correlation = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]

    return lag < 0