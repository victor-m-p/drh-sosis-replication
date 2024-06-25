"""
Type of violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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

from helper_functions import code_conflict

conflict_order = [
    "No Violent Conflict",
    "Internal Only",
    "Internal and External",
    "External Only",
]

palette = sns.color_palette("tab10", n_colors=4)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
sns.set_style("white")
for i, variable in enumerate(variable_dict.keys()):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    # subset and code conflict
    wide_subset = answers_wide[[variable, "violent_external", "violent_internal"]]
    wide_subset = wide_subset.dropna()
    wide_subset["conflict_type"] = wide_subset.apply(code_conflict, axis=1)
    # get counts for plot
    group_counts = wide_subset.groupby("conflict_type").size()
    # plot
    sns.barplot(
        x="conflict_type",
        y=variable,
        data=wide_subset,
        ax=ax,
        order=conflict_order,
        palette=palette,
        label=conflict_order,
    )
    # only show y label for first column
    if col == 0:
        ax.set_ylabel("Fraction Yes", fontsize=18)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"n={group_counts[label]}" for label in conflict_order],
        fontsize=14,
    )
    ax.set_title(f"{variable_dict.get(variable)}", fontsize=16)

# maximally 2 decimal places for y-axis
formatter = ticker.FormatStrFormatter("%.2f")
for ax in axes.flat:
    ax.yaxis.set_major_formatter(formatter)

# create single legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=len(conflict_order),
    frameon=False,
    fontsize=16,
)
plt.tight_layout()
plt.savefig("../figures/all_markers_overall.pdf", bbox_inches="tight")
plt.savefig("../figures/png/all_markers_overall.png", bbox_inches="tight", dpi=300)
