"""
External violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")
answers.groupby(["question_id", "question_short"]).size()

# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# take out relevant columns for now
wide_subset = answers_wide[
    ["circumcision", "tattoos/scarification", "permanent scarring", "violent external"]
]
# only drop nan values for the violent external column
wide_subset = wide_subset.dropna(subset=["violent external"])

# collapse groups into has external vs. does not have external
from helper_functions import code_conflict_collapsed

wide_subset["conflict_type_collapsed"] = wide_subset.apply(
    code_conflict_collapsed, axis=1
)
wide_subset = wide_subset.drop("violent external", axis=1)
wide_subset["entry_id"] = wide_subset.index

df_long = pd.melt(
    wide_subset,
    id_vars=["entry_id", "conflict_type_collapsed"],
    var_name="marker",
    value_name="value",
)

# now we can drop nan values
df_long = df_long.dropna()

# run statistical tests
from scipy.stats import chi2_contingency


def run_chi2_test(df, marker):
    marker_df = df[df["marker"] == marker]
    marker_df["value"] = marker_df["value"].astype(int)
    contingency_table = pd.crosstab(
        marker_df["value"], marker_df["conflict_type_collapsed"]
    )
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p


markers = ["circumcision", "tattoos/scarification", "permanent scarring"]
chi2_results = {marker: run_chi2_test(df_long, marker) for marker in markers}

# Prepare labels
labels = [
    (
        f"{marker}\n(χ²={result[0]:.2f}; p<{result[1]:.2f})"
        if result[1] < 0.05
        else f"{marker}\n(χ²={result[0]:.2f}; ns)"
    )
    for marker, result in chi2_results.items()
]

# Assuming the data transformation is done
plt.figure(figsize=(7, 4))
sns.set_style("white")
bar_plot = sns.barplot(
    data=df_long,
    x="marker",
    y="value",
    hue="conflict_type_collapsed",
    # palette="gray",
    errorbar=("ci", 95),  # straight confidence intervals of mean
)

# Add annotations
bar_plot.set_xticklabels(labels, fontsize=12)

# Finalize the plot
plt.title(
    ""
)  # ("Frequencies of Permanent Markers by Presence and Absence of External Warfare")
plt.ylabel("Fraction Yes", fontsize=14)
plt.xlabel("")
plt.legend(title="", loc="upper right")
plt.ylim(0, 0.5)
plt.savefig("../figures/external_warfare_markers.png", bbox_inches="tight", dpi=300)
plt.savefig("../figures/external_warfare_markers.pdf", bbox_inches="tight")
