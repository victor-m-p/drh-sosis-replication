"""
External violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

answers = pd.read_csv("../data/preprocessed/answers.csv")
answers.groupby(["question_id", "question_short"]).size()

# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# take out relevant columns for now
wide_subset = answers_wide[["extra-ritual in-group markers", "violent external"]]
wide_subset = wide_subset.dropna(
    subset=["extra-ritual in-group markers", "violent external"]
)

# collapse groups into has external vs. does not have external
from helper_functions import code_conflict_collapsed

wide_subset["conflict_type_collapsed"] = wide_subset.apply(
    code_conflict_collapsed, axis=1
)

wide_sorted = (
    wide_subset.groupby("conflict_type_collapsed", as_index=False)[
        "extra-ritual in-group markers"
    ]
    .mean()
    .sort_values("extra-ritual in-group markers")
)

group_counts = wide_subset.groupby("conflict_type_collapsed").size()

fig, ax = plt.subplots(figsize=(6, 4))
sns.set_style("white")
sns.barplot(
    x="conflict_type_collapsed",
    y="extra-ritual in-group markers",
    data=wide_subset,
    ax=ax,
    order=wide_sorted["conflict_type_collapsed"],
)
ax.set_xlabel("")
ax.set_xticklabels(
    [
        f"{label}\n(n={group_counts[label]})"
        for label in wide_sorted["conflict_type_collapsed"]
    ],
    fontsize=14,
)
ax.set_ylabel("Fraction Yes", fontsize=16)
ax.set_title("")  # ("Extra-ritual in-group markers by conflict type")
plt.savefig(
    "../figures/extra_ritual_by_conflict_collapsed.png", bbox_inches="tight", dpi=300
)
plt.savefig("../figures/extra_ritual_by_conflict_collapsed.pdf", bbox_inches="tight")
