library(progress)

causality_truefalse <- function(result, causality_type = "entropy") {
  
  result_array <- vector("logical", length = length(result))
  
  pb <- progress_bar$new(total = length(result_array), clear = FALSE, width = 80)
  
  for (i in seq_along(result_array)) {
    pb$tick()
    
    if (!is.null((result[[i]])$XgCsY_trns)) {
      result_array[i] <- (result[[i]])$XgCsY_trns
    } else {
      result_array[i] <- FALSE  
    }
  }
  return(result_array)
}

ConfusionMatrix <- function(positive,negative) {
  
  positive_table <- table(positive)
  negative_table <- table(negative)
  
  TP <- 0
  TN <- 0
  FP <- 0
  FN <- 0
  
  if ("TRUE" %in% names(positive_table)) {
    TP <- positive_table[["TRUE"]]
  }
  if ("FALSE" %in% names(negative_table)) {
    TN <- negative_table[["FALSE"]]
  }
  if ("TRUE" %in% names(negative_table)) {
    FP <- negative_table[["TRUE"]]
  }
  if ("FALSE" %in% names(positive_table)) {
    FN <- positive_table[["FALSE"]]
  }
  
  f1_score <- function(TP, FP, FN) {
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    f1 <- 2 * (precision * recall) / (precision + recall)
    return(c(precision, recall, f1))
  }
  
  scores <- f1_score(TP, FP, FN)
  precision <- scores[1]
  recall <- scores[2]
  f1 <- scores[3]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  
  return(list(TP = TP, TN = TN, FP = FP, FN = FN, Precision = precision, Recall = recall, F1 = f1, Accuracy = accuracy))
}

ResultDataframe <- function(version="short"){
  
  dataset1 <- ConfusionMatrix(extracted_positive_entropy_1,extracted_negative_entropy_1)
  dataset2 <- ConfusionMatrix(extracted_positive_entropy_2,extracted_negative_entropy_2)
  dataset3 <- ConfusionMatrix(extracted_positive_entropy_3,extracted_negative_entropy_3)
  
  if (version=="short"){
    ResultFrame <-data.frame(
      Dataset = c("Dataset1", "Dataset2", "Dataset3", "Dataset1+2+3"),
      Precision = c(dataset1[["Precision"]],
                    dataset2[["Precision"]],
                    dataset3[["Precision"]],
                    (dataset1[["Precision"]]+dataset2[["Precision"]]+dataset3[["Precision"]])/3),
      Recall = c(dataset1[["Recall"]],
                 dataset2[["Recall"]],
                 dataset3[["Recall"]],
                 (dataset1[["Recall"]]+dataset2[["Recall"]]+dataset3[["Recall"]])/3),
      F1 = c(dataset1[["F1"]],
             dataset2[["F1"]],
             dataset3[["F1"]],
             (dataset1[["F1"]]+dataset2[["F1"]]+dataset3[["F1"]])/3),
      Accuracy = c(dataset1[["Accuracy"]],
                   dataset2[["Accuracy"]],
                   dataset3[["Accuracy"]],
                   (dataset1[["Accuracy"]]+dataset2[["Accuracy"]]+dataset3[["Accuracy"]])/3)
    )
  }else{
    ResultFrame <-data.frame(
      Dataset = c("Dataset1", "Dataset2", "Dataset3", "Dataset1+2+3"),
      TP = c(dataset1[["TP"]],
             dataset2[["TP"]],
             dataset3[["TP"]],
             (dataset1[["TP"]]+dataset2[["TP"]]+dataset3[["TP"]])),
      TN = c(dataset1[["TN"]],
             dataset2[["TN"]],
             dataset3[["TN"]],
             (dataset1[["TN"]]+dataset2[["TN"]]+dataset3[["TN"]])),
      FP = c(dataset1[["FP"]],
             dataset2[["FP"]],
             dataset3[["FP"]],
             (dataset1[["FP"]]+dataset2[["FP"]]+dataset3[["FP"]])),
      FN = c(dataset1[["FN"]],
             dataset2[["FN"]],
             dataset3[["FN"]],
             (dataset1[["FN"]]+dataset2[["FN"]]+dataset3[["FN"]])),
      Precision = c(dataset1[["Precision"]],
                    dataset2[["Precision"]],
                    dataset3[["Precision"]],
                    (dataset1[["Precision"]]+dataset2[["Precision"]]+dataset3[["Precision"]])/3),
      Recall = c(dataset1[["Recall"]],
                 dataset2[["Recall"]],
                 dataset3[["Recall"]],
                (dataset1[["Recall"]]+dataset2[["Recall"]]+dataset3[["Recall"]])/3),
      F1 = c(dataset1[["F1"]],
             dataset2[["F1"]],
             dataset3[["F1"]],
            (dataset1[["F1"]]+dataset2[["F1"]]+dataset3[["F1"]])/3),
      Accuracy = c(dataset1[["Accuracy"]],
                   dataset2[["Accuracy"]],
                   dataset3[["Accuracy"]],
                  (dataset1[["Accuracy"]]+dataset2[["Accuracy"]]+dataset3[["Accuracy"]])/3)
      )
  }
  
  return(ResultFrame)
}

#----------------------------------------------------------------

file_bases <- c("negative_entropy1", "negative_entropy2", "negative_entropy3", 
                "positive_entropy1", "positive_entropy2", "positive_entropy3")

for (base in file_bases) {
  file_path <- paste0("result/", base, ".RData")
  load(file_path)
  assign(base, entropy_result)
}

#----------------------------------------------------------------

extracted_negative_entropy_1 <- causality_truefalse(negative_entropy1)
extracted_negative_entropy_2 <- causality_truefalse(negative_entropy2)
extracted_negative_entropy_3 <- causality_truefalse(negative_entropy3)

extracted_positive_entropy_1 <- causality_truefalse(positive_entropy1)
extracted_positive_entropy_2 <- causality_truefalse(positive_entropy2)
extracted_positive_entropy_3 <- causality_truefalse(positive_entropy3)

#----------------------------------------------------------------

print(ResultDataframe())
