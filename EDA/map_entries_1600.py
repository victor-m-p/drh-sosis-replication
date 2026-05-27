"""
vmp 2026-05-27
World map of DRH entries colored by violent_external × marker, split pre/post 1600.
Two PNGs per marker (pre-1600 and post-1600) saved to figures/maps_1600/.
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

os.makedirs("../figures/maps_1600", exist_ok=True)

# ── load geography once ────────────────────────────────────────────────────────
regions = pd.read_csv("../data/raw/region_data.csv")[["region_id", "gis_region"]].drop_duplicates("region_id")
regions = gpd.GeoDataFrame(regions, geometry=gpd.GeoSeries.from_wkt(regions["gis_region"]), crs="EPSG:4326")
regions["centroid"] = regions.geometry.centroid

entries = pd.read_csv("../data/raw/entry_data.csv")[["entry_id", "region_id", "year_from", "year_to"]]
entries["year_mid"] = (entries["year_from"] + entries["year_to"]) / 2

world = gpd.read_file(get_path("naturalearth.land"))

# ── one pair of maps per marker ────────────────────────────────────────────────
for marker in MARKERS:
    df = pd.read_csv(f"../data/model/external/input/{marker}.csv")[
        ["entry_id", "violent_external", marker]
    ]
    df = df.merge(entries[["entry_id", "region_id", "year_mid"]], on="entry_id", how="left")
    df = df.merge(regions[["region_id", "centroid"]], on="region_id", how="left")
    df = gpd.GeoDataFrame(df, geometry="centroid", crs="EPSG:4326")
    df = df[df.geometry.x.between(-180, 180) & df.geometry.y.between(-90, 90)]

    df["group"] = (df["violent_external"].map({1: "Warfare", 0: "No warfare"})
                   + ", "
                   + df[marker].map({1: "Marker", 0: "No marker"}))

    for period_label, mask in [("pre-1600", df["year_mid"] < 1600),
                                ("post-1600", df["year_mid"] >= 1600)]:
        subset_df = df[mask]
        n_total = len(subset_df)

        fig, ax = plt.subplots(figsize=(16, 8))
        world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)

        for label, color in COLORS.items():
            subset = subset_df[subset_df["group"] == label]
            subset.plot(ax=ax, color=color, markersize=25, alpha=0.7,
                        label=f"{label} (n={len(subset)})", marker="o")

        ax.set_title(f"{marker} — {period_label} (n={n_total})", fontsize=14, pad=8)
        ax.legend(loc="lower left", fontsize=16, markerscale=2.0, framealpha=0.8)
        ax.set_axis_off()
        plt.tight_layout()
        fname = f"../figures/maps_1600/{marker}_{period_label.replace('-', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {fname}")
