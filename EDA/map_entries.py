"""
vmp 2026-05-13
World map of DRH entries colored by violent_external × marker.
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

# ── load geography once ────────────────────────────────────────────────────────
regions = pd.read_csv("../data/raw/region_data.csv")[["region_id", "gis_region"]].drop_duplicates("region_id")
regions = gpd.GeoDataFrame(regions, geometry=gpd.GeoSeries.from_wkt(regions["gis_region"]), crs="EPSG:4326")
regions["centroid"] = regions.geometry.centroid

entry_region = pd.read_csv("../data/raw/entry_data.csv")[["entry_id", "region_id"]]

world = gpd.read_file(get_path("naturalearth.land"))

# ── one map per marker ─────────────────────────────────────────────────────────
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

    ax.legend(loc="lower left", fontsize=16, markerscale=2.0, framealpha=0.8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"../figures/maps/{marker}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {marker}.png")
