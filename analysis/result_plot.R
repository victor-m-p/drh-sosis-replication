library(pacman)
p_load(tidyverse, bayesplot)

# find the relevant files; 
directory_path = ""
file_paths <- list.files(path = directory_path, pattern = "\\.csv$", full.names = TRUE)
