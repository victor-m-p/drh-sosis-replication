"""
vmp 2024-05-07
correct entry metadata and code regions 
"""

# imports
import pandas as pd
import geopandas as gpd

# load data
entrydata = pd.read_csv("../data/raw/entry_data.csv")
entrydata = entrydata[["entry_id", "entry_name", "year_from", "year_to", "region_id"]]


# 1. fix dates
# this should be removed once we have the clean dump.
corrected_dates = {
    # entry_id, year_from, year_to
    1570: [1900, 2024],
    1777: [1881, 1939],
    1779: [1956, 2023],
    1905: [1514, 1575],
    1907: [1425, 1505],
    1928: [1793, 1864],
    2016: [909, 1171],
    2032: [1767, 1828],
    2040: [1752, 1804],
    2048: [777, 909],
    2075: [1214, 1273],
    2087: [839, 923],
    2147: [1229, 1574],
    2236: [935, 1012],
    2245: [652, 1240],
    2258: [817, 870],
    2272: [1058, 1111],
}

for entry_id, (year_from, year_to) in corrected_dates.items():
    entrydata.loc[entrydata["entry_id"] == entry_id, ["year_from", "year_to"]] = [
        year_from,
        year_to,
    ]

# also include regions
region_data = pd.read_csv("../data/raw/region_data.csv")
region_data = region_data[["region_id", "world_region"]].drop_duplicates()
entrydata = entrydata.merge(region_data, on="region_id", how="left")
entrydata.to_csv("../data/preprocessed/entry_data.csv", index=False)
