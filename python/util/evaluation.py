import numpy as np 
from sklearn.metrics import confusion_matrix

import sys
import os 

sys.path.append(os.path.abspath(os.path.join('..', '.')))

from following_motif_method.followingMotif import following_relation_method
from following_motif_method.compared_method import FLICA,max_correlation


def ground_truth(ts):
    gt_ts = ts[1]
    ground_truth = []
    for i in range(len(gt_ts[1])):
        ground_truth = ground_truth + list(range(gt_ts[0][i],gt_ts[1][i]+1))

    return np.array(ground_truth)

def Non(a_list):
    return np.setdiff1d(np.array(range(len(a_list)+1)), a_list)

def confusion_matrix(TP, TN, FP, FN):

    def f1_score(TP, TN, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, TN, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy

def evaluate_a_pair_ts(leader,follower,seed=0,wd=30,gap=10):
    ground_truth_lead    = ground_truth(leader) 
    ground_truth_follow  = ground_truth(follower)

    ground_truth_lead = ground_truth_lead[:len(ground_truth_follow)]

    lead_ts   = (leader[0]).copy()
    follow_ts = (follower[0]).copy()

    np.random.seed(seed)
    result = following_relation_method(lead_ts,follow_ts,wd=wd,gap=gap)
    print(fr"{result[1]}")

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

def confusion_matrix(cm_result):
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

if __name__ == "__main__":
    print("hugh")