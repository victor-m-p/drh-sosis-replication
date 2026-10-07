"""
vmp 2026-05-13
World map of entries in the external violent conflict analysis, colored by
violent_external x marker, with eHRAF-sourced entries circled.
One PDF per marker saved to data/model/external/maps/ (tattoos_scarification is
Figure 4 in the paper).
Points are region representative points (entries sharing a region overlap).
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
OUT = "../data/model/external/maps"
os.makedirs(OUT, exist_ok=True)

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
    # 11.5 in wide gives a map just wider than the 3-column legend at fontsize 16
    fig, ax = plt.subplots(figsize=(11.5, 6))
    world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)
    for label, color in COLORS.items():
        subset = df[df["group"] == label]
        subset.plot(ax=ax, color=color, markersize=25, alpha=0.7,
                    label=f"{label} (n={len(subset)})", marker="o")
    ehraf = df[df["is_ehraf"] == True]
    if len(ehraf):
        ehraf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.2,
                   markersize=50, alpha=0.9, marker="o", label=f"eHRAF (n={len(ehraf)})")
    # legend below the map so it never covers entries (e.g. southern South America),
    # stretched to exactly the map width
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0, 1, 0), mode="expand", ncol=3,
              fontsize=16, markerscale=2.0, frameon=False, borderaxespad=0)
    ax.set_xlim(-180, 180)  # no side padding, so the land spans the full legend width
    ax.set_ylim(-60, 85)  # crop Antarctica, which has no entries
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{OUT}/external_map_{marker}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
