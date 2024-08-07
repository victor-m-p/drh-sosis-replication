"""
VMP 2024-07-31
Same as markers_external.py, but without eHRAF entries.
Not directly reported. We focus on the Bayesian analysis without eHRAF entries in the SI.
Similary analysis run for all_markers_internal.py without eHRAF entries.
No effects significant (like in the main analysis). 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

pd.options.mode.chained_assignment = None  # default='warn'

# load
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")

# remove HRAF
entries = pd.read_csv("../data/preprocessed/entries_clean.csv")
entries_not_hraf = entries[entries["data_source"] != "eHRAF"]
answers = answers[answers["entry_id"].isin(entries_not_hraf["entry_id"])]

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

from helper_functions import code_external_conflict
from helper_functions import run_chi2_test

conflict_order = ["No External Violent Conflict", "External Violent Conflict"]

palette = sns.color_palette("tab10", n_colors=4)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
sns.set_style("white")
for i, variable in enumerate(variable_dict.keys()):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    # subset and code conflict
    wide_subset = answers_wide[[variable, "violent_external"]]
    wide_subset = wide_subset.dropna()
    # collapse groups into has external vs. does not have external
    wide_subset["conflict_type"] = wide_subset.apply(code_external_conflict, axis=1)
    wide_subset["entry_id"] = wide_subset.index
    df_long = pd.melt(
        wide_subset,
        id_vars=["entry_id", "conflict_type"],
        var_name="marker",
        value_name="value",
    )
    # get counts for plot
    group_counts = wide_subset.groupby("conflict_type").size()
    # now we can drop nan values
    df_long = df_long.dropna()

    # run statistical tests
    from helper_functions import run_chi2_test

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
    ax.set_title(f"{variable_dict.get(variable)}\n({labels[0]})", fontsize=16)

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
plt.savefig("../figures/markers_external_not_hraf.pdf", bbox_inches="tight")
plt.savefig(
    "../figures/png/markers_external_not_hraf.png", bbox_inches="tight", dpi=300
)
