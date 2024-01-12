__all__ = ['ground_truth', 'Non','evaluate_a_pair_ts','calculate_matrix','confusion_matrix_many','sklearn_cm_extract']

import numpy as np 
from following_motif_relation.followingMotif import following_motif_method

def ground_truth(ts):
    gt_ts = ts[1]
    ground_truth = []
    for i in range(len(gt_ts[1])):
        ground_truth = ground_truth + list(range(gt_ts[0][i],gt_ts[1][i]+1))

    return np.array(ground_truth)

def Non(a_list):
    return np.setdiff1d(np.array(range(len(a_list)+1)), a_list)

def calculate_matrix(TP, TN, FP, FN):
    def f1_score(TP, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy

def evaluate_a_pair_ts(leader,follower,wd=30,thres=10,randseed=0,verbose=True):
    ground_truth_lead    = ground_truth(leader) 
    ground_truth_follow  = ground_truth(follower)

    ground_truth_lead = ground_truth_lead[:len(ground_truth_follow)]

    lead_ts   = (leader[0]).copy()
    follow_ts = (follower[0]).copy()

    np.random.seed(randseed)
    result = following_motif_method(lead_ts,follow_ts,wd=wd,thres=thres,randseed=randseed)
    if verbose:
        print(f'leadVol: {result[1]}')

    new_index_lead   = list(set(result[2][1]))
    new_index_follow = list(set(result[2][0]))

    new_index_lead = new_index_lead[:len(new_index_follow)]

    #==============================================

    # TP => #Point Method following match  #Point GT following
    existence_lead_TP = np.isin(new_index_lead, ground_truth_lead)
    existence_follow_TP = np.isin(new_index_follow, ground_truth_follow)
    TP = np.sum(existence_lead_TP)+np.sum(existence_follow_TP)
    
    # TN => #Point Method Non-following match  #Point GT Non-following
    existence_lead_TN = np.isin(Non(new_index_lead), Non(ground_truth_lead))
    existence_follow_TN = np.isin(Non(new_index_follow), Non(ground_truth_follow))
    TN = np.sum(existence_lead_TN)+np.sum(existence_follow_TN)
   
    # FP => #Point Method following match  #Point GT NON-following
    existence_lead_FP = np.isin(new_index_lead, Non(ground_truth_lead))
    existence_follow_FP = np.isin(new_index_follow, Non(ground_truth_follow))
    FP = np.sum(existence_lead_FP)+np.sum(existence_follow_FP)

    # FN => #Point Method NON-following NOT match #Point GT following
    existence_lead_FN = np.isin(Non(new_index_lead), ground_truth_lead)
    existence_follow_FN = np.isin(Non(new_index_follow), ground_truth_follow)
    FN = np.sum(existence_lead_FN)+np.sum(existence_follow_FN)

    #==============================================

    data = [TP, TN, FP, FN]

    #==============================================

    return np.array(data)

def confusion_matrix_TF(cm_result):
    TP = cm_result[1, 1]
    TN = cm_result[0, 0]
    FP = cm_result[0, 1]
    FN = cm_result[1, 0]

    def f1_score(TP, TN, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, TN, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy

def confusion_matrix_many(cm_result1,cm_result2,cm_result3):
    TP = cm_result1[1, 1]+cm_result2[1, 1]+cm_result3[1, 1]
    TN = cm_result1[0, 0]+cm_result2[0, 0]+cm_result3[0, 0]
    FP = cm_result1[0, 1]+cm_result2[0, 1]+cm_result3[0, 1]
    FN = cm_result1[1, 0]+cm_result2[1, 0]+cm_result3[1, 0]

    def f1_score(TP, TN, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, TN, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy

def sklearn_cm_extract(cm_result):
    TP = cm_result[1, 1]
    TN = cm_result[0, 0]
    FP = cm_result[0, 1]
    FN = cm_result[1, 0]

    return TP,TN,FP,FN

if __name__ == "__main__":
    print("hugh")