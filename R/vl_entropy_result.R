library(progress)
library(VLTimeCausality)

vl_entropy_result <- function(follower_list, leader_list, file_name, entropy_type = "positive") {

  pb <- progress_bar$new(total = length(follower_list), clear = FALSE, width = 80)
  
  entropy_result <- vector("list", length = length(follower_list))
  for (i in seq_along(follower_list)) {
    pb$tick()  
    Y_vector <- follower_list[[i]]$X0
    X_vector <- leader_list[[i]]$X0
    
    if (entropy_type == "positive") {
      entropy_result[[i]] <- VLTransferEntropy(Y = Y_vector, X = X_vector)
    } else if (entropy_type == "negative") {
      entropy_result[[i]] <- VLTransferEntropy(Y = X_vector, X = Y_vector)
    }
  }
  
  save(entropy_result, file = paste0(file_name, ".RData"))
  return(entropy_result)
}

params <- list(
  positive1 = list(follower_list = datasets[["follower_list_1"]], leader_list = datasets[["leader_list_1"]], file_name = "positive_entropy1", entropy_type = "positive"),
  positive2 = list(follower_list = datasets[["follower_list_2"]], leader_list = datasets[["leader_list_2"]], file_name = "positive_entropy2", entropy_type = "positive"),
  positive3 = list(follower_list = datasets[["follower_list_3"]], leader_list = datasets[["leader_list_3"]], file_name = "positive_entropy3", entropy_type = "positive"),
  negative1 = list(follower_list = datasets[["follower_list_1"]], leader_list = datasets[["leader_list_1"]], file_name = "negative_entropy1", entropy_type = "negative"),
  negative2 = list(follower_list = datasets[["follower_list_2"]], leader_list = datasets[["leader_list_2"]], file_name = "negative_entropy2", entropy_type = "negative"),
  negative3 = list(follower_list = datasets[["follower_list_3"]], leader_list = datasets[["leader_list_3"]], file_name = "negative_entropy3", entropy_type = "negative")
)

entropy_results <- lapply(params, function(p) {
  vl_entropy_result(p$follower_list, p$leader_list, p$file_name, p$entropy_type)
})
