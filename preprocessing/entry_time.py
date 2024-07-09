"""
vmp 2024-06-25
Gather entry data (year and region)
"""

# imports
import pandas as pd
import geopandas as gpd

# load data
entrydata = pd.read_csv("../data/raw/entry_data.csv")
entrydata = entrydata[
    ["entry_id", "entry_name", "year_from", "year_to", "region_id", "data_source"]
]

# also include regions
region_data = pd.read_csv("../data/raw/region_data.csv")
region_data = region_data[["region_id", "world_region"]].drop_duplicates()
entrydata = entrydata.merge(region_data, on="region_id", how="left")
entrydata.to_csv("../data/preprocessed/entries_clean.csv", index=False)
