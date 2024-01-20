load_dataset <- function(folder, i, leadfollow, size = 1000) {
  path_pattern <- paste0(folder, "/",leadfollow, "_list_",i,"_", 0:(size-1), ".csv")
  lapply(path_pattern, function(file) {
    read.csv(file)
  })
}

load_timeseries <- function() {
  datasets <- list()
  for (i in 1:3) {
    folder <- paste0("R/dataset/dataset", i)
    datasets[[paste0("leader_list_", i)]]   <- load_dataset(folder, i, "leader", 1000)
    datasets[[paste0("follower_list_", i)]] <- load_dataset(folder, i, "follower", 1000)
  }
  return(datasets)
}

#print(dir.exists(folder))
datasets <- load_timeseries()