import numpy as np 
import matrixProfile

def following_relation_method(lead_ts,follow_ts,wd=300,gap=10):

    following_motif_index = [None,None]

    leader_mp,_   = matrixProfile.stamp(follow_ts,wd,lead_ts)
    follower_mp,_ = matrixProfile.stamp(lead_ts,wd,follow_ts)

    rmp = [follower_mp,leader_mp]

    for i in range(2):

        motif_percentile = 50-gap 
        
        motif_index = rmp[i] <np.percentile(rmp[i], motif_percentile)
        motif_index_mp = np.where(motif_index)[0] 

        following_motif_index[i] = motif_index_mp.copy()
        following_motif_index[i] = list(set(following_motif_index[i]))

    following_motif_index[1] = following_motif_index[1][:len(following_motif_index[0])]

    
    lead_result_value = np.mean(np.array(following_motif_index[0])-np.array(following_motif_index[1]))
    lead_result       = lead_result_value>0
    

    return lead_result,lead_result_value,following_motif_index


