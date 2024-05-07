import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import pygeos

# Countries and World Regions
countries_gdf = gpd.read_file("countries.gpkg")
world_regions = pd.read_csv("../data/raw/world_regions.csv")
countries_regions = countries_gdf.merge(world_regions, on="iso_a3", how="left")

# If no associated world region we cannot use it
countries_regions = countries_regions.dropna()
countries_regions = countries_regions.rename(
    {"Country": "country", "World.Region": "world_region"}, axis=1
)

# Load the gis data
entrydata = pd.read_csv("../data/raw/entry_data.csv")
entrydata = entrydata[entrydata["geom_completed"] == True]
entrydata = entrydata[entrydata["geom"].notnull()]
entrydata = entrydata[["entry_id", "entry_name", "region_id", "geom"]]
entrydata["geometry"] = entrydata["geom"].apply(wkt.loads)
entry_gdf = gpd.GeoDataFrame(entrydata, geometry="geometry")
entry_gdf = entry_gdf.set_crs("EPSG:4326")

# find intersections
entry_gdf["entry_id"].nunique()  # n=1617
intersection = gpd.overlay(countries_regions, entry_gdf, how="intersection")
intersection["entry_id"].nunique()  # n=1588

# get mode world region for overlapping cases
entry_wr_intersection_mode = (
    intersection.groupby(["entry_id", "entry_name"])["world_region"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)

# fix cases with no overlap (find nearest)
non_overlapping = entry_gdf[~entry_gdf["entry_id"].isin(intersection["entry_id"])]
countries_regions_pygeos = pygeos.from_shapely(countries_regions["geometry"])
non_overlapping_pygeos = pygeos.from_shapely(non_overlapping["geometry"])

non_overlapping["id"] = np.nan
for idx, geometry in enumerate(non_overlapping_pygeos):
    distances = pygeos.distance(geometry, countries_regions_pygeos)
    closest_country_idx = np.argmin(distances)
    non_overlapping.iloc[idx, non_overlapping.columns.get_loc("id")] = (
        closest_country_idx
    )

non_overlapping["id"] = non_overlapping["id"].astype(int)
non_overlapping = non_overlapping[
    ["entry_id", "entry_name", "region_id", "id"]
].drop_duplicates()

# merge on countries regions
non_overlapping = non_overlapping.merge(countries_regions, on="id", how="left")
non_overlapping = non_overlapping[["entry_id", "entry_name", "world_region"]]

# concatenate with the intersection data
entry_world_region = pd.concat([entry_wr_intersection_mode, non_overlapping])
entry_world_region["entry_id"].nunique()  # 1617

# add back the raw geom information
entrydata = entrydata[["entry_id", "region_id", "geom"]].drop_duplicates()
entry_world_region = entry_world_region.merge(entrydata, on="entry_id", how="inner")
entry_world_region = entry_world_region.sort_values("entry_id")

entry_world_region
# save data
entry_world_region.to_csv("../data/preprocessed/entry_regions.csv", index=False)
