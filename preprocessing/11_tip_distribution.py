"""
Plots for understanding the distribution of tattoos_scarification data.
Saved to preprocessing/figures/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.makedirs("figures", exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
data = pd.read_csv("../data/phylo_input/tattoos_scarification.csv")

N_TIPS = 10
top_tips = data["tip_name"].value_counts().head(N_TIPS).index.tolist()
df = data[data["tip_name"].isin(top_tips)].copy()

# ── aggregate per tip ─────────────────────────────────────────────────────────
agg = (df.groupby("tip_name")
         .agg(
             n                          = ("entry_id",                    "count"),
             prop_violent               = ("violent_external",             "mean"),
             prop_ritual                = ("tattoos_scarification",         "mean"),
         )
         .loc[top_tips]   # keep the original frequency order
         .reset_index())

# short label: last two components of tip name (e.g. "ROMANCE.LATIN")
agg["label"] = agg["tip_name"].apply(lambda x: ".".join(x.split(".")[-2:]))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
fig.suptitle(
    f"Top {N_TIPS} most common tips — violent conflict & tattoos/scarification",
    fontsize=12, fontweight="bold")

VARS = [
    ("prop_violent",  "violent_external",      "#DD8452", axes[0]),
    ("prop_ritual",   "tattoos_scarification", "#4C72B0", axes[1]),
]

for prop_col, raw_col, colour, ax in VARS:
    y = np.arange(len(agg))

    # proportion bar
    ax.barh(y, agg[prop_col], color=colour, alpha=0.55, height=0.5)

    # individual observations as jittered dots
    for i, tip in enumerate(top_tips):
        vals = df.loc[df["tip_name"] == tip, raw_col].values
        jitter = np.random.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(vals + np.random.uniform(-0.01, 0.01, len(vals)),
                   i + jitter,
                   color=colour, alpha=0.6, s=18, zorder=3)

    # n labels
    for i, row in agg.iterrows():
        ax.text(1.03, list(agg.index).index(i),
                f"n={int(row['n'])}",
                va="center", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["label"], fontsize=9)
    ax.set_xlim(-0.08, 1.15)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Proportion coded 1", fontsize=9)
    ax.set_title(raw_col.replace("_", " "), fontsize=10)
    ax.axvline(0.5, color="grey", lw=0.8, linestyle="--")
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/top_tips_violent_ritual.png", dpi=150, bbox_inches="tight")
plt.show()

# ── plot 2: distribution of observations per tip ─────────────────────────────
tip_counts = data["tip_name"].value_counts()

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.hist(tip_counts.values, bins=range(1, tip_counts.max() + 2),
         color="#4C72B0", edgecolor="white", align="left")
ax2.set_xlabel("Observations per tip", fontsize=10)
ax2.set_ylabel("Number of tips", fontsize=10)
plt.tight_layout()
plt.savefig("figures/tip_count_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── plot 4: proportion violent external over time (100-year bins) ─────────────
entries = pd.read_csv("../data/preprocessed/entries_clean.csv",
                      usecols=["entry_id", "year_from", "year_to"])
entries["year_mid"] = (entries["year_from"] + entries["year_to"]) / 2

timed = data.merge(entries, on="entry_id", how="left").dropna(subset=["year_mid"])

BIN = 500
timed["bin"] = (np.floor(timed["year_mid"] / BIN) * BIN).astype(int)
binned = (timed.groupby("bin")["violent_external"]
               .agg(prop="mean", n="count")
               .reset_index())

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.bar(binned["bin"], binned["prop"], width=BIN * 0.85,
        color="#DD8452", edgecolor="white", align="edge")
ax4.set_xlabel("Year (CE/BCE)", fontsize=10)
ax4.set_ylabel("Proportion violent external", fontsize=10)
ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig("figures/violent_over_time.png", dpi=150, bbox_inches="tight")
plt.show()

# ── plot 3a: entries per world region, stacked by violent_external ────────────
regions = (data.groupby("world_region")["entry_id"]
               .count()
               .sort_values(ascending=False)
               .index.tolist())

C_VIOLENT     = "#DD8452"
C_NON_VIOLENT = "#4C72B0"

def stacked_region_bar(ax, groupby_col, label0, label1, title):
    rc = (data.groupby(["world_region", groupby_col])
              .size()
              .reset_index(name="n"))
    bottoms = np.zeros(len(regions))
    for val, colour, lbl in [(0, C_NON_VIOLENT, label0), (1, C_VIOLENT, label1)]:
        vals = []
        for region in regions:
            subset = rc[(rc["world_region"] == region) & (rc[groupby_col] == val)]
            vals.append(subset["n"].values[0] if len(subset) else 0)
        vals = np.array(vals)
        ax.bar(regions, vals, bottom=bottoms, color=colour, label=lbl,
               edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xlabel("World region", fontsize=10)
    ax.set_ylabel("Number of entries", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=9)

fig3, ax3 = plt.subplots(figsize=(9, 5))
stacked_region_bar(ax3, "violent_external",
                   "violent external = 0", "violent external = 1",
                   "Entries per world region (coloured by violent external)")
plt.tight_layout()
plt.savefig("figures/region_violent_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── plot 3b: entries per world region, stacked by tattoos_scarification ───────
fig3b, ax3b = plt.subplots(figsize=(9, 5))
stacked_region_bar(ax3b, "tattoos_scarification",
                   "tattoos/scarification = 0", "tattoos/scarification = 1",
                   "Entries per world region (coloured by tattoos/scarification)")
plt.tight_layout()
plt.savefig("figures/region_outcome_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
