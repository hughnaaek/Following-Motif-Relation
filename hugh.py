def calculate_matrix(cm_result):
    TP = cm_result[1, 1]
    TN = cm_result[0, 0]
    FP = cm_result[0, 1]
    FN = cm_result[1, 0]

    def f1_score(TP, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy

def calculate_matrix(TP, TN, FP, FN):
    def f1_score(TP, FP, FN):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return precision,recall,f1
    
    precision,recall,f1 = f1_score(TP, TN, FP, FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    return TP,TN,FP,FN,precision,recall,f1,accuracy