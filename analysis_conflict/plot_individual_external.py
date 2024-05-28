"""
External violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")
answers.groupby(["question_id", "question_short"]).size()

# dependent variable, figtitle, filename
dependent_variable_list = [
    (
        "extra-ritual in-group markers",
        "Extra-ritual in-group markers",
        "markers_external",
    ),
    ("archaic ritual language", "Archaic ritual language", "archaic_external"),
]

# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# loop
for variable in dependent_variable_list:
    dependent_variable = variable[0]
    dependent_figtitle = variable[1]
    dependent_filename = variable[2]

    # take out relevant columns for now
    wide_subset = answers_wide[
        [
            dependent_variable,
            "violent external",
        ]
    ]
    # only drop nan values for the violent external column
    wide_subset = wide_subset.dropna(subset=["violent external"])

    # collapse groups into has external vs. does not have external
    from helper_functions import code_conflict_collapsed

    wide_subset["conflict_type"] = wide_subset.apply(code_conflict_collapsed, axis=1)
    wide_subset = wide_subset.drop("violent external", axis=1)
    wide_subset["entry_id"] = wide_subset.index

    df_long = pd.melt(
        wide_subset,
        id_vars=["entry_id", "conflict_type"],
        var_name="marker",
        value_name="value",
    )

    # now we can drop nan values
    df_long = df_long.dropna()

    # run statistical tests
    from helper_functions import run_chi2_test

    chi2, pval = run_chi2_test(df_long, dependent_variable)

    # Prepare labels
    labels = [
        (
            f"{dependent_figtitle}\n(χ²={chi2:.2f}; p<0.05)"
            if pval < 0.05
            else f"{dependent_figtitle}\n(χ²={chi2:.2f}; ns)"
        )
    ]

    # Assuming the data transformation is done
    plt.figure(figsize=(6, 6))
    sns.set_style("white")
    bar_plot = sns.barplot(
        data=df_long,
        x="marker",
        y="value",
        hue="conflict_type",
        # palette="gray",
        errorbar=("ci", 95),  # straight confidence intervals of mean
    )

    # Add annotations
    bar_plot.set_xticklabels(labels, fontsize=12)

    # Finalize the plot
    plt.title("")  # f"{dependent_figtitle} by External Violent Conflict", fontsize=16
    plt.ylabel("Fraction Yes", fontsize=14)
    plt.xlabel("")
    plt.legend(title="", loc="lower right", fontsize=12)
    plt.savefig(f"../figures/{dependent_filename}.pdf", bbox_inches="tight")
    plt.savefig(
        f"../figures/png/{dependent_filename}.png", bbox_inches="tight", dpi=300
    )
