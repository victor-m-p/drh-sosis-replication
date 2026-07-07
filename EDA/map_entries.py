"""
vmp 2026-05-13
World map of DRH entries colored by violent_external x marker.
One PNG per marker saved to figures/maps/.
Points are region centroids (entries sharing a region overlap — add jitter if needed).
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geodatasets import get_path

MARKERS = [
    "circumcision", "dress", "extra_ritual_group_markers", "food_taboos",
    "hair", "ornaments", "permanent_scarring", "tattoos_scarification",
]
COLORS = {
    "No warfare, No marker": "#4393c3",
    "No warfare, Marker":    "#f4a582",
    "Warfare, No marker":    "#92c5de",
    "Warfare, Marker":       "#d6604d",
}
os.makedirs("../figures/maps", exist_ok=True)

# load region information
regions = pd.read_csv("../data/raw/region_data.csv")[["region_id", "gis_region"]].drop_duplicates("region_id")
regions = gpd.GeoDataFrame(regions, geometry=gpd.GeoSeries.from_wkt(regions["gis_region"]), crs="EPSG:4326")

# Compute a representative point per region, in a projected (equal-area) CRS.
# We use representative_point() rather than centroid(): several DRH regions are
# multi-part or ring-shaped (e.g. a Mediterranean coastal strip), and a plain
# centroid can fall outside the polygon entirely (e.g. in open water). 
# representative_point() is guaranteed to fall within the polygon.
regions_proj = regions.to_crs("ESRI:54009")  # World Mollweide, equal-area
regions["centroid"] = gpd.GeoSeries(
    regions_proj.geometry.representative_point(), crs="ESRI:54009"
).to_crs("EPSG:4326")

entry_data = pd.read_csv("../data/raw/entry_data.csv")[["entry_id", "region_id", "data_source"]]
entry_data["is_ehraf"] = entry_data["data_source"] == "eHRAF"
entry_region = entry_data[["entry_id", "region_id", "is_ehraf"]]
world = gpd.read_file(get_path("naturalearth.land"))

for marker in MARKERS:
    df = pd.read_csv(f"../data/model/external/input/{marker}.csv")[
        ["entry_id", "violent_external", marker]
    ]
    df = df.merge(entry_region, on="entry_id", how="left")
    df = df.merge(regions[["region_id", "centroid"]], on="region_id", how="left")
    df = gpd.GeoDataFrame(df, geometry="centroid", crs="EPSG:4326")
    df = df[df.geometry.x.between(-180, 180) & df.geometry.y.between(-90, 90)]
    df["group"] = (df["violent_external"].map({1: "Warfare", 0: "No warfare"})
                   + ", "
                   + df[marker].map({1: "Marker", 0: "No marker"}))
    fig, ax = plt.subplots(figsize=(16, 8))
    world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)
    for label, color in COLORS.items():
        subset = df[df["group"] == label]
        subset.plot(ax=ax, color=color, markersize=25, alpha=0.7,
                    label=f"{label} (n={len(subset)})", marker="o")
    ehraf = df[df["is_ehraf"] == True]
    if len(ehraf):
        ehraf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2,
                   markersize=50, alpha=0.9, marker="o", label=f"eHRAF (n={len(ehraf)})")
    ax.legend(loc="lower left", fontsize=16, markerscale=2.0, framealpha=0.8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"../figures/maps/{marker}.pdf", bbox_inches="tight")
    plt.close()

'''
Trouble-shoot region plots:
The problem I believe is that we have some entries that cross the 
anti-meridian. For instance Entry ID 871.
'''

# quick check on entries that are on/off land.
land = world.union_all() 
df["on_land"] = df.geometry.within(land)

# quick plotting function.
def plot_entry_check(entry_id, df, regions, world, buffer_deg=5):
    """
    Plot a single entry's centroid/representative point against the world map,
    zoomed in, with the source region polygon overlaid for context.
    """
    row = df[df["entry_id"] == entry_id]
    if row.empty:
        print(f"entry_id {entry_id} not found in df")
        return

    region_id = row["region_id"].values[0]
    point = row.geometry.values[0]

    # pull the actual region polygon (not just the point) for context
    region_poly = regions[regions["region_id"] == region_id]

    fig, ax = plt.subplots(figsize=(10, 10))
    world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)

    if not region_poly.empty:
        region_poly.set_geometry(
            gpd.GeoSeries.from_wkt(region_poly["gis_region"]), crs="EPSG:4326"
        ).plot(ax=ax, color="none", edgecolor="darkgreen", linewidth=1.5, alpha=0.8)

    row.plot(ax=ax, color="red", markersize=80, marker="o", zorder=5)

    ax.set_xlim(point.x - buffer_deg, point.x + buffer_deg)
    ax.set_ylim(point.y - buffer_deg, point.y + buffer_deg)
    ax.set_title(f"entry_id={entry_id}, region_id={region_id}")
    plt.show()


# this shows the problem we have.
plot_entry_check(871, df, regions, world, buffer_deg=200)