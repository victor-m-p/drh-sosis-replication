import numpy as np
import pandas as pd

# load
entry_regions = pd.read_csv("../data/preprocessed/entry_regions.csv")
answerset = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# select and merge
entry_regions = entry_regions[["entry_id", "entry_name", "world_region"]]
answerset = answerset.merge(
    entry_regions, on="entry_id", how="inner"
)  # some we do not have

# check overall n in each region
entry_regions = answerset[["entry_id", "entry_name", "world_region"]].drop_duplicates()
region_counts = entry_regions.groupby("world_region").size()


# test a simple model with time as a predictor as well:
# ...
