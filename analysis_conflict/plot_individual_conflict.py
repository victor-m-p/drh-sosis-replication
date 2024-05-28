"""
Type of violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# dependent variable, figtitle, filename
dependent_variable_list = [
    (
        "extra-ritual in-group markers",
        "Extra-ritual in-group markers",
        "markers_conflict",
    ),
    ("archaic ritual language", "Archaic ritual language", "archaic_conflict"),
]

# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# loop over variables
for variable in dependent_variable_list:
    dependent_variable = variable[0]
    dependent_figtitle = variable[1]
    dependent_filename = variable[2]

    wide_subset = answers_wide[
        [dependent_variable, "violent external", "violent internal"]
    ]
    wide_subset = wide_subset.dropna()  # drops around half

    # code warfare
    from helper_functions import code_conflict

    wide_subset["conflict_type"] = wide_subset.apply(code_conflict, axis=1)

    # for sorting the plot
    wide_sorted = (
        wide_subset.groupby("conflict_type", as_index=False)[dependent_variable]
        .mean()
        .sort_values(dependent_variable)
    )

    # Get number of observations for the plot
    group_counts = wide_subset.groupby("conflict_type").size()

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.set_style("white")
    sns.barplot(
        x="conflict_type",
        y=dependent_variable,
        data=wide_subset,
        ax=ax,
        order=wide_sorted["conflict_type"],
    )
    ax.set_xlabel("")
    ax.set_xticklabels(
        [
            f"{label}\n(n={group_counts[label]})"
            for label in wide_sorted["conflict_type"]
        ],
        fontsize=12,
    )
    ax.set_ylabel("Fraction Yes", fontsize=14)
    ax.set_title(f"{dependent_figtitle} by conflict type", fontsize=16)
    plt.savefig(f"../figures/{dependent_filename}.pdf", bbox_inches="tight")
    plt.savefig(
        f"../figures/png/{dependent_filename}.png", bbox_inches="tight", dpi=300
    )
