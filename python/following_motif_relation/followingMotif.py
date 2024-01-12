__all__ = ['following_motif_method']

from .matrixProfile import *
import numpy as np 

def following_motif_method(lead_ts,follow_ts,wd=300,thres=10,randseed=0):
    np.random.seed(randseed)

    following_motif_index = [None,None]

    leader_mp,_   = stamp(follow_ts,wd,lead_ts)
    follower_mp,_ = stamp(lead_ts,wd,follow_ts)

    rmp = [follower_mp,leader_mp]

    for i in range(2):

        motif_percentile = 50-thres 
        
        motif_index = rmp[i] <np.percentile(rmp[i], motif_percentile)
        motif_index_mp = np.where(motif_index)[0] 

        following_motif_index[i] = motif_index_mp.copy()
        following_motif_index[i] = list(set(following_motif_index[i]))

    following_motif_index[1] = following_motif_index[1][:len(following_motif_index[0])]

    lead_index   = list(following_motif_index[1])
    follow_index = list(following_motif_index[0])

    max_leader   = len(leader_mp)
    max_follower = len(follower_mp)
    normalized_x = lambda x,max : (x - 0) / (max - 0)

    lead_index_norm   = [normalized_x(x,max_leader) for x in lead_index] 
    follow_index_norm = [normalized_x(x,max_follower) for x in follow_index]

    lead_result_value = np.mean(np.array(follow_index_norm)-np.array(lead_index_norm))
    lead_result       = lead_result_value>0
    

    return lead_result,lead_result_value,following_motif_index


