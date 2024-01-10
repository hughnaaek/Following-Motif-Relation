# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

#from . import distanceProfile
#from . import order
#from .utils import mass, movmeanstd
import numpy as np
import multiprocessing
from functools import partial
import math

#from .scrimp import scrimp_plus_plus
#---------------------------------------------------------------
class Order:
    """
    An object that defines the order in which the distance profiles are calculated for a given Matrix Profile
    """
    def next(self):
        raise NotImplementedError("next() not implemented")

class linearOrder(Order):
    """
    An object that defines a linear (iterative) order in which the distance profiles are calculated for a given Matrix Profile
    """
    def __init__(self,m):
        self.m = m
        self.idx = -1

    def next(self):
        """
        Advances the Order object to the next index
        """
        self.idx += 1
        if self.idx < self.m:
            return self.idx
        else:
            return None


class randomOrder(Order):
    """
    An object that defines a random order in which the distance profiles are calculated for a given Matrix Profile
    """
    def __init__(self,m, random_state=None):
        self.idx = -1
        self.indices = np.arange(m)
        self.random_state = random_state
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        np.random.shuffle(self.indices)

    def next(self):
        """
        Advances the Order object to the next index
        """
        self.idx += 1
        try:
            return self.indices[self.idx]

        except IndexError:
            return None

#---------------------------------------------------------------

#---------------------------------------------------------------

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
import numpy.fft as fft

def zNormalize(ts):
    """
    Returns a z-normalized version of a time series.

    Parameters
    ----------
    ts: Time series to be normalized
    """

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")
    else:
        ts /= std

    return ts

def zNormalizeEuclidian(tsA,tsB):
    """
    Returns the z-normalized Euclidian distance between two time series.

    Parameters
    ----------
    tsA: Time series #1
    tsB: Time series #2
    """

    if len(tsA) != len(tsB):
        raise ValueError("tsA and tsB must be the same length")

    return np.linalg.norm(zNormalize(tsA.astype("float64")) - zNormalize(tsB.astype("float64")))

def movmeanstd(ts,m):
    """
    Calculate the mean and standard deviation within a moving window passing across a time series.

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    movmean = segSum/m
    movstd = np.sqrt(segSumSq / m - (segSum/m) ** 2)

    return [movmean,movstd]

def movstd(ts,m):
    """
    Calculate the standard deviation within a moving window passing across a time series.

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    return np.sqrt(segSumSq / m - (segSum/m) ** 2)

def slidingDotProduct(query,ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.

    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """

    m = len(query)
    n = len(ts)


    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]


    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))


    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]

def DotProductStomp(ts,m,dot_first,dot_prev,order):
    """
    Updates the sliding dot product for a time series ts from the previous dot product dot_prev.

    Parameters
    ----------
    ts: Time series under analysis.
    m: Length of query within sliding dot product.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    order: The location of the first point in the query.
    """

    l = len(ts)-m+1
    dot = np.roll(dot_prev,1)

    dot += ts[order+m-1]*ts[m-1:l+m]-ts[order-1]*np.roll(ts[:l],1)

    #Update the first value in the dot product array
    dot[0] = dot_first[order]

    return dot


def mass(query,ts):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.

    Parameters
    ----------
    :query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    :ts: Time series to compare against query.
    """

    #query_normalized = zNormalize(np.copy(query))
    m = len(query)
    q_mean = np.mean(query)
    q_std = np.std(query)
    mean, std = movmeanstd(ts,m)
    dot = slidingDotProduct(query,ts)

    #res = np.sqrt(2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std)))
    res = 2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std))


    return res

def massStomp(query,ts,dot_first,dot_prev,index,mean,std):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries using the STOMP dot product speedup. Note that we are returning the square of MASS.

    Parameters
    ----------
    query: Time series snippet to evaluate. Note that, for STOMP, the query must be a subset of ts.
    ts: Time series to compare against query.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    index: The location of the first point in the query.
    mean: Array containing the mean of every subsequence in ts.
    std: Array containing the standard deviation of every subsequence in ts.
    """
    m = len(query)
    dot = DotProductStomp(ts,m,dot_first,dot_prev,index)

    #Return both the MASS calcuation and the dot product
    res = 2*m*(1-(dot-m*mean[index]*mean)/(m*std[index]*std))

    return res, dot


def apply_av(mp,av=[1.0]):
    """
    Applies an annotation vector to a Matrix Profile.

    Parameters
    ----------
    mp: Tuple containing the Matrix Profile and Matrix Profile Index.
    av: Numpy array containing the annotation vector.
    """

    if len(mp[0]) != len(av):
        raise ValueError(
            "Annotation Vector must be the same length as the matrix profile")
    else:
        av_max = np.max(av)
        av_min = np.min(av)
        if av_max > 1 or av_min < 0:
            raise ValueError("Annotation Vector must be between 0 and 1")
        mp_corrected = mp[0] + (1 - np.array(av)) * np.max(mp[0])
        return (mp_corrected, mp[1])


def is_self_join(tsA, tsB):
    """
    Helper function to determine if a self join is occurring or not. When tsA 
    is absolutely equal to tsB, a self join is occurring.

    Parameters
    ----------
    tsA: Primary time series.
    tsB: Subquery time series.
    """
    return tsB is None or np.array_equal(tsA, tsB)


#---------------------------------------------------------------

def naiveDistanceProfile(tsA,idx,m,tsB = None):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the naive all-pairs comparison.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx: (idx+m)]
    distanceProfile = []
    n = len(tsB)

    for i in range(n-m+1):
        distanceProfile.append(zNormalizeEuclidian(query,tsB[i:i+m]))

    dp = np.array(distanceProfile)

    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))

        dp[trivialMatchRange[0]: trivialMatchRange[1]] = np.inf

    return (dp,np.full(n-m+1,idx,dtype=float))


def massDistanceProfile(tsA,idx,m,tsB = None):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the more efficient MASS comparison.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """

    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx:(idx+m)]
    n = len(tsB)
    distanceProfile = np.real(np.sqrt(mass(query,tsB).astype(complex)))
    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))
        distanceProfile[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    #Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile,np.full(n-m+1,idx,dtype=float))

def mass_distance_profile_parallel(indices, tsA=None, tsB=None, m=None):
    """
    Computes distance profiles for the given indices either via self join or similarity search.

    Parameters
    ----------
    indices: Array of indices to compute distance profile for.
    tsA: Time series containing the query for which to calculate the distance profile.
    tsB: Time series to compare the query against. Note that, for the time being, only tsB = tsA is allowed
    m: Length of query.
    """
    distance_profiles = []

    for index in indices:
        distance_profiles.append(massDistanceProfile(tsA, index, m, tsB=tsB))

    return distance_profiles


def STOMPDistanceProfile(tsA,idx,m,tsB,dot_first,dp,mean,std):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the even more efficient iterative STOMP calculation. Note that the method requires a pre-calculated 'initial' sliding dot product.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, for the time being, only tsB = tsA is allowed
    dot_first: The 'initial' sliding dot product, or QT(1,1) in Zhu et.al
    dp: The dot product between tsA and the query starting at index m-1
    mean: Array containing the mean of every subsequence of length m in tsA (moving window)
    std: Array containing the mean of every subsequence of length m in tsA (moving window)
    """

    selfJoin = is_self_join(tsA, tsB)
    if selfJoin:
        tsB = tsA

    query = tsA[idx:(idx+m)]
    n = len(tsB)

    #Calculate the first distance profile via MASS
    if idx == 0:
        distanceProfile = np.real(np.sqrt(mass(query,tsB).astype(complex)))

        #Currently re-calculating the dot product separately as opposed to updating all of the mass function...
        dot = slidingDotProduct(query,tsB)

    #Calculate all subsequent distance profiles using the STOMP dot product shortcut
    else:
        res, dot = massStomp(query,tsB,dot_first,dp,idx,mean,std)
        distanceProfile = np.real(np.sqrt(res.astype(complex)))


    if selfJoin:
        trivialMatchRange = (int(max(0,idx - np.round(m/2,0))),int(min(idx + np.round(m/2+1,0),n)))
        distanceProfile[trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

    #Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile,np.full(n-m+1,idx,dtype=float)), dot

#---------------------------------------------------------------------

def is_array_like(a):
    """
    Helper function to determine if a value is array like.
    Parameters
    ----------
    a : obj
        Object to test.
    Returns
    -------
    True or false respectively.
    """
    return isinstance(a, tuple([list, tuple, np.ndarray]))

def to_np_array(a):
    """
    Helper function to convert tuple or list to np.ndarray.
    Parameters
    ----------
    a : Tuple, list or np.ndarray
        The object to transform.
    Returns
    -------
    The np.ndarray.
    Raises
    ------
    ValueError
        If a is not a valid type.
    """
    if not is_array_like(a):
        raise ValueError('Unable to convert to np.ndarray!')

    return np.array(a)

def _clean_nan_inf(ts):
    """
    Converts tuples & lists to Numpy arrays and replaces nan and inf values with zeros

    Parameters
    ----------
    ts: Time series to clean
    """

    #Convert time series to a Numpy array
    ts = to_np_array(ts)

    search = (np.isinf(ts) | np.isnan(ts))
    ts[search] = 0

    return ts


def _self_join_or_not_preprocess(tsA, tsB, m):
    """
    Core method for determining if a self join is occuring and returns appropriate
    profile and index numpy arrays with correct dimensions as all np.nan values.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    tsB: Time series to compare the query against. Note that, if no value is provided, ts_b = ts_a by default.
    m: Length of subsequence to compare.
    """
    n = len(tsA)
    if tsB is not None:
        n = len(tsB)

    shape = n - m + 1

    return (np.full(shape, np.inf), np.full(shape, np.inf))

def _matrixProfile(tsA,m,orderClass,distanceProfileFunction,tsB=None):
    """
    Core method for calculating the Matrix Profile

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    orderClass: Method defining the order in which distance profiles are calculated.
    distanceProfileFunction: Method for calculating individual distance profiles.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile.
    """

    order = orderClass(len(tsA)-m+1)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    if not is_array_like(tsB):
        tsB = tsA

    tsA = _clean_nan_inf(tsA)
    tsB = _clean_nan_inf(tsB)

    idx=order.next()
    while idx != None:
        (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA,idx,m,tsB)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

    return (mp,mpIndex)

def _stamp_parallel(tsA, m, tsB=None, sampling=0.2, n_threads=-1, random_state=None):
    """
    Computes distance profiles in parallel using all CPU cores by default.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile. 0 to 1
    n_threads: Number of threads to use in parallel mode. Defaults to using all CPU cores.
    random_state: Set the random seed generator for reproducible results.
    """
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()

    n = len(tsA)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    if not is_array_like(tsB):
        tsB = tsA

    tsA = _clean_nan_inf(tsA)
    tsB = _clean_nan_inf(tsB)

    # determine sampling size
    sample_size = math.ceil((n - m + 1) * sampling)

    # generate indices to sample and split based on n_threads
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n - m + 1)
    indices = np.random.choice(indices, size=sample_size, replace=False)
    indices = np.array_split(indices, n_threads)

    # create pool of workers and compute
    with multiprocessing.Pool(processes=n_threads) as pool:
        func = partial(mass_distance_profile_parallel, tsA=tsA, tsB=tsB, m=m)
        results = pool.map(func, indices)

    # The overall matrix profile is the element-wise minimum of each sub-profile, and each element of the overall
    # matrix profile index is the time series position of the corresponding sub-profile.
    for result in results:
        for dp, querySegmentsID in result:
            #Check which of the indices have found a new minimum
            idsToUpdate = dp < mp

            #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
            mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

            #Update the matrix profile to include the new minimum values (where appropriate)
            mp = np.minimum(mp, dp)

    return (mp, mpIndex)

def _matrixProfile_sampling(tsA,m,orderClass,distanceProfileFunction,tsB=None,sampling=0.2,random_state=None):
    order = orderClass(len(tsA)-m+1, random_state=random_state)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    if not is_array_like(tsB):
        tsB = tsA

    tsA = _clean_nan_inf(tsA)
    tsB = _clean_nan_inf(tsB)

    idx=order.next()

    #Define max numbers of iterations to sample
    iters = (len(tsA)-m+1)*sampling

    iter_val = 0

    while iter_val < iters:
        (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA,idx,m,tsB)

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        iter_val += 1
    return (mp,mpIndex)


#Write matrix profile function for STOMP and then consolidate later! (aka link to the previous distance profile)
def _matrixProfile_stomp(tsA,m,orderClass,distanceProfileFunction,tsB=None):
    order = orderClass(len(tsA)-m+1)
    mp, mpIndex = _self_join_or_not_preprocess(tsA, tsB, m)

    if not is_array_like(tsB):
        tsB = tsA

    tsA = _clean_nan_inf(tsA)
    tsB = _clean_nan_inf(tsB)

    idx=order.next()

    #Get moving mean and standard deviation
    mean, std = movmeanstd(tsA,m)

    #Initialize code to set dot_prev to None for the first pass
    dp = None

    #Initialize dot_first to None for the first pass
    dot_first = None

    while idx != None:

        #Need to pass in the previous sliding dot product for subsequent distance profile calculations
        (distanceProfile,querySegmentsID),dot_prev = distanceProfileFunction(tsA,idx,m,tsB,dot_first,dp,mean,std)

        if idx == 0:
            dot_first = dot_prev

        #Check which of the indices have found a new minimum
        idsToUpdate = distanceProfile < mp

        #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mpIndex[idsToUpdate] = querySegmentsID[idsToUpdate]

        #Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp,distanceProfile)
        idx = order.next()

        dp = dot_prev
    return (mp,mpIndex)

def stampi_update(tsA,m,mp,mpIndex,newval,tsB=None,distanceProfileFunction=massDistanceProfile):
    '''Updates the self-matched matrix profile for a time series TsA with the arrival of a new data point newval. Note that comparison of two separate time-series with new data arriving will be built later -> currently, tsB should be set to tsA'''

    #Update time-series array with recent value
    tsA_new = np.append(np.copy(tsA),newval)

    #Expand matrix profile and matrix profile index to include space for latest point
    mp_new= np.append(np.copy(mp),np.inf)
    mpIndex_new = np.append(np.copy(mpIndex),np.inf)

    #Determine new index value
    idx = len(tsA_new)-m

    (distanceProfile,querySegmentsID) = distanceProfileFunction(tsA_new,idx,m,tsB)

    #Check which of the indices have found a new minimum
    idsToUpdate = distanceProfile < mp_new

    #Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
    mpIndex_new[idsToUpdate] = querySegmentsID[idsToUpdate]

    #Update the matrix profile to include the new minimum values (where appropriate)
    mp_final = np.minimum(np.copy(mp_new),distanceProfile)

    #Finally, set the last value in the matrix profile to the minimum of the distance profile (with corresponding index)
    mp_final[-1] = np.min(distanceProfile)
    mpIndex_new[-1] = np.argmin(distanceProfile)

    return (mp_final,mpIndex_new)


def naiveMP(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the naive all-pairs calculation.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile(tsA,m,linearOrder,naiveDistanceProfile,tsB)

def stmp(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed linearly across every time series index.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile(tsA,m,linearOrder,massDistanceProfile,tsB)

def stamp(tsA,m,tsB=None,sampling=0.2, n_threads=None, random_state=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed in a random order.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    sampling: The percentage of all possible distance profiles to sample for the final Matrix Profile. 0 to 1
    n_threads: Number of threads to use in parallel mode. Defaults to single threaded mode. Set to -1 to use all threads.
    random_state: Set the random seed generator for reproducible results.
    """
    if sampling > 1 or sampling < 0:
        raise ValueError('Sampling value must be a percentage in decimal format from 0 to 1.')

    if n_threads is None:
        return _matrixProfile_sampling(tsA,m,randomOrder,massDistanceProfile,tsB,sampling=sampling,random_state=random_state)

    return _stamp_parallel(tsA, m, tsB=tsB, sampling=sampling, n_threads=n_threads, random_state=random_state)

def stomp(tsA,m,tsB=None):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed according to the directed STOMP procedure.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    tsB: Time series to compare the query against. Note that, if no value is provided, tsB = tsA by default.
    """
    return _matrixProfile_stomp(tsA,m,linearOrder,STOMPDistanceProfile,tsB)



if __name__ == "__main__":
    import doctest
    doctest.method()
