"""
VMP 2026-01-31 (updated.)
Figure generated in this script is not included in the paper. 
Claim that internal conflict does not drive markers (transitory or permanent) based on Chi-Square statistics here.
Emphasis is on transitory markers, but we also do not observe an effect on permanent markers. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from helper_functions import code_internal_conflict, run_chi2_test

pd.options.mode.chained_assignment = None  # default='warn'

# load
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")

# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# variable list
variable_dict = {
    # super questions
    "permanent_scarring": "Permanent Scarring",
    "extra_ritual_group_markers": "Extra Ritual In-Group Markers",
    # permanent markers
    "circumcision": "Circumcision",
    "tattoos_scarification": "Tattoos or Scarification",
    # transitory markers
    "dress": "Dress",
    "food_taboos": "Food Taboos",
    "hair": "Hair",
    "ornaments": "Ornaments",
}

# setup
conflict_order = ["No Internal Conflict", "Internal Conflict only"]
palette = sns.color_palette("tab10", n_colors=2)

# plot 
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
sns.set_style("white")

# loop
for i, variable in enumerate(variable_dict.keys()):
    # for plot
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    
    # subset and code conflict
    wide_subset = answers_wide[[variable, "violent_external", "violent_internal"]]
    wide_subset = wide_subset.dropna()
    
    # collapse groups into has external vs. does not have external
    wide_subset["conflict_type"] = wide_subset.apply(code_internal_conflict, axis=1)
    wide_subset["entry_id"] = wide_subset.index

    # melt for chi2
    df_long = pd.melt(
        wide_subset,
        id_vars=["entry_id", "conflict_type"],
        var_name="marker",
        value_name="value",
    )
    df_long = df_long[df_long["conflict_type"].isin(conflict_order)]
    
    # get counts for plot
    group_counts = wide_subset.groupby("conflict_type").size()
    
    # now we can drop nan values
    df_long = df_long.dropna()

    # run statistics (same as in first docu.)
    chi2, pval = run_chi2_test(df_long, variable)

    # Prepare labels
    labels = [(f"χ²={chi2:.2f}; p<0.05" if pval < 0.05 else f"χ²={chi2:.2f}; ns")]

    # plot
    sns.barplot(
        x="conflict_type",
        y=variable,
        data=wide_subset,
        ax=ax,
        order=conflict_order,
        palette=palette,
        hue="conflict_type",
        hue_order=conflict_order,
        legend=False,
    )
    
    # remove 
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    # set xticks to sample size
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"n={group_counts[label]}" for label in conflict_order],
        fontsize=18,
    )
    
    # title per plot (marker)
    ax.set_title(f"{variable_dict.get(variable)}\n({labels[0]})", fontsize=16)

# shared y label (figure text)
fig.text(
    -0.02,
    0.5,
    "Fraction of markers present",
    va="center",
    rotation="vertical",
    fontdict={"fontsize": 20, "fontweight": "light"},
)

# maximally 2 decimal places for y-axis
formatter = ticker.FormatStrFormatter("%.2f")
for ax in axes.flat:
    ax.yaxis.set_major_formatter(formatter)

# create single legend
legend_handles = [
    Patch(facecolor=palette[0], label=conflict_order[0]),
    Patch(facecolor=palette[1], label=conflict_order[1]),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
    fontsize=20,
)
plt.tight_layout()
plt.savefig("../figures/markers_internal.pdf", bbox_inches="tight")
plt.savefig("../figures/png/markers_internal.png", bbox_inches="tight", dpi=300)
