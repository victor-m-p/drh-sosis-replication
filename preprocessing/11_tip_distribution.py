"""
Exploratory plot: distribution of violent_external and extra_ritual_group_markers
for the N most common tips. Run interactively; no output saved.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── load ──────────────────────────────────────────────────────────────────────
data = pd.read_csv("../data/phylo_input/extra_ritual_group_markers.csv")

N_TIPS = 10
top_tips = data["tip_name"].value_counts().head(N_TIPS).index.tolist()
df = data[data["tip_name"].isin(top_tips)].copy()

# ── aggregate per tip ─────────────────────────────────────────────────────────
agg = (df.groupby("tip_name")
         .agg(
             n                          = ("entry_id",                    "count"),
             prop_violent               = ("violent_external",             "mean"),
             prop_ritual                = ("extra_ritual_group_markers",   "mean"),
         )
         .loc[top_tips]   # keep the original frequency order
         .reset_index())

# short label: last two components of tip name (e.g. "ROMANCE.LATIN")
agg["label"] = agg["tip_name"].apply(lambda x: ".".join(x.split(".")[-2:]))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
fig.suptitle(
    f"Top {N_TIPS} most common tips — violent conflict & ritual markers",
    fontsize=12, fontweight="bold")

VARS = [
    ("prop_violent",  "violent_external",           "#DD8452", axes[0]),
    ("prop_ritual",   "extra_ritual_group_markers",  "#4C72B0", axes[1]),
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
plt.show()

# ── plot 2: distribution of observations per tip (all tips) ───────────────────
tip_counts = data["tip_name"].value_counts()

fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
fig2.suptitle("Distribution of observations per tip (all tips, extra ritual)",
              fontsize=12, fontweight="bold")

# left: histogram of counts
ax = axes2[0]
ax.hist(tip_counts.values, bins=range(1, tip_counts.max() + 2),
        color="#4C72B0", edgecolor="white", align="left")
ax.set_xlabel("Observations per tip", fontsize=10)
ax.set_ylabel("Number of tips", fontsize=10)
ax.set_title("How many tips have n=1, n=2, …?", fontsize=10)
ax.axvline(tip_counts.median(), color="red", lw=1.2, linestyle="--",
           label=f"median = {tip_counts.median():.0f}")
ax.legend(fontsize=9)

# annotate key numbers
for n_val, label in [(1, "n=1"), (2, "n=2")]:
    count = (tip_counts == n_val).sum()
    ax.text(n_val, count + 0.3, f"{count} tips", ha="center",
            fontsize=8, color="#333333")

# right: cumulative — what share of observations come from tips with n ≥ k?
ax2 = axes2[1]
sorted_counts = tip_counts.sort_values(ascending=False).to_numpy(dtype=float)
cum_obs   = np.cumsum(sorted_counts)
cum_share = cum_obs / cum_obs[-1]
ax2.plot(np.arange(1, len(sorted_counts) + 1), cum_share,
         color="#4C72B0", lw=2)
ax2.set_xlabel("Number of tips (sorted by size)", fontsize=10)
ax2.set_ylabel("Cumulative share of observations", fontsize=10)
ax2.set_title("How concentrated are observations across tips?", fontsize=10)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax2.axhline(0.5, color="red", lw=1, linestyle="--", label="50% of obs")
ax2.legend(fontsize=9)

print(f"Total tips:          {len(tip_counts)}")
print(f"Tips with n=1:       {(tip_counts == 1).sum()}")
print(f"Tips with n≥2:       {(tip_counts >= 2).sum()}")
print(f"Tips with n≥5:       {(tip_counts >= 5).sum()}")
print(f"Median obs per tip:  {tip_counts.median():.0f}")
print(f"Max obs per tip:     {tip_counts.max()} ({tip_counts.idxmax()})")

plt.tight_layout()
plt.show()
