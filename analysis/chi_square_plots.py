"""
vmp 2026-05-13
Chi-square tests and bar plots for all four analyses (SI).
Reads from data/model/{analysis}/input/ — one CSV per marker.
Output: figures/chi-square/{analysis}.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency

ANALYSES = {
    "external":         ("violent_external", "No External Conflict", "External Conflict"),
    "external_noeHRAF": ("violent_external", "No External Conflict", "External Conflict"),
    "internal":         ("violent_internal", "No Internal Conflict", "Internal Conflict"),
    "internal_noeHRAF": ("violent_internal", "No Internal Conflict", "Internal Conflict"),
}

MARKERS = {
    "circumcision":               "Circumcision",
    "dress":                      "Dress",
    "extra_ritual_group_markers": "Extra-Ritual In-Group Markers",
    "food_taboos":                "Food Taboos",
    "hair":                       "Hair",
    "ornaments":                  "Ornaments",
    "permanent_scarring":         "Permanent Scarring",
    "tattoos_scarification":      "Tattoos or Scarification",
}

os.makedirs("../figures/chi-square", exist_ok=True)
palette = sns.color_palette("tab10", n_colors=2)

for analysis, (predictor, label_0, label_1) in ANALYSES.items():

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    sns.set_style("white")

    for i, (marker, marker_label) in enumerate(MARKERS.items()):
        ax = axes[i // 4, i % 4]

        df = pd.read_csv(f"../data/model/{analysis}/input/{marker}.csv")

        # chi-square test
        ct = pd.crosstab(df[marker], df[predictor])
        chi2, p, _, _ = chi2_contingency(ct, correction=False)
        stat_label = f"χ²={chi2:.2f}; p<0.05" if p < 0.05 else f"χ²={chi2:.2f}; ns"

        # bar plot: proportion of marker present by conflict group
        sns.barplot(
            x=predictor, y=marker, data=df,
            order=[0, 1], palette=palette,
            hue=predictor, hue_order=[0, 1], legend=False,
            ax=ax,
        )

        counts = df.groupby(predictor)[marker].count()
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"n={counts[0]}", f"n={counts[1]}"], fontsize=14)
        ax.set_title(f"{marker_label}\n({stat_label})", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.text(-0.02, 0.5, "Fraction of markers present", va="center",
             rotation="vertical", fontdict={"fontsize": 16, "fontweight": "light"})

    legend_handles = [
        Patch(facecolor=palette[0], label=label_0),
        Patch(facecolor=palette[1], label=label_1),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(f"../figures/chi-square/{analysis}.png", bbox_inches="tight", dpi=300)
    plt.close()

