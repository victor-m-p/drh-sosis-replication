"""
vmp 2024-05-07
correct entry metadata and code regions 
"""

# imports
import pandas as pd
import geopandas as gpd

# load data
entrydata = pd.read_csv("../data/raw/entry_data.csv")

# 1. fix dates
corrected_dates = {
    # entry_id, year_from, year_to
    1905: [1514, 1575],
    1907: [1425, 1595],
    1928: [1793, 1864],
    2032: [1767, 1828],
    2040: [1752, 1894],
}

for entry_id, (year_from, year_to) in corrected_dates.items():
    entrydata.loc[entrydata["entry_id"] == entry_id, ["year_from", "year_to"]] = [
        year_from,
        year_to,
    ]

entrydata = entrydata[["entry_id", "entry_name", "year_from", "year_to"]]
entrydata.to_csv("../data/preprocessed/entry_time.csv", index=False)

# 2. code regions
# Countries and World Regions
countries_gdf = gpd.read_file("countries.gpkg")
world_regions = pd.read_csv("../data/raw/world_regions.csv")
countries_regions = countries_gdf.merge(world_regions, on="iso_a3", how="left")

### Wrangle our previous GIS data
gis_metadata = pd.read_csv("../data/preprocessed/gis_metadata.csv")
entry_metadata = pd.read_csv("../data/preprocessed/entry_metadata.csv")
entry_metadata = entry_metadata.merge(gis_metadata, on="region_id", how="inner")
entry_metadata = entry_metadata[
    ["entry_id", "entry_name", "region_id", "geometry"]
].drop_duplicates()
entry_gis_complete = entry_metadata[entry_metadata["geometry"].notnull()]
entry_gis_complete["geometry"] = entry_gis_complete["geometry"].apply(wkt.loads)
entry_gis_gdf = gpd.GeoDataFrame(entry_gis_complete, geometry="geometry")
entry_gis_gdf = entry_gis_gdf.set_crs("EPSG:4326")

# 2. get the area associated with gis regions
from kml_functions import split_regions, calculate_gis_area

### NB: we should split this process into 2 steps:
### 1. get the meridian splits and save this
### 2. get the areas and save them.

assert len(gis_regions) == gis_regions["region_id"].nunique()
region_split_antimeridian = split_regions(gis_regions)
region_area = calculate_gis_area(region_split_antimeridian)
region_split_antimeridian.to_csv("../data/preprocessed/gis_metadata.csv", index=False)

# why do they become so big?
region_area = region_area[["region_id", "region_area", "region_type"]].drop_duplicates()

# insert nan for region area where there is no gis_region
region_area = region_area.groupby("region_id")["region_area"].sum()
region_area = region_area.reset_index(name="count")
region_area = region_area.rename(columns={"count": "region_area"})
region_overall = mode_region.merge(region_area, on="region_id", how="left")
