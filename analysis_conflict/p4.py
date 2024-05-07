"""
Trying to create composite score.
Not really convinced that this is useful. 

Not sure whether to include both the super-question (extra-ritual)
and the sub-questions. 
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# try with just composite of what we already have;
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)
answers_wide["entry_id"] = answers_wide.index

# assign groups
from helper_functions import code_conflict

answers_wide = answers_wide.dropna(subset=["violent external", "violent internal"])
answers_wide["conflict_type"] = answers_wide.apply(code_conflict, axis=1)

# take out relevant columns
answers_wide = answers_wide[
    [
        "circumcision",
        "tattoos/scarification",
        "permanent scarring",
        "conflict_type",
        "extra-ritual in-group markers",
    ]
].dropna()


# only if they have answers for all of these
answers_wide["composite_score"] = answers_wide[
    [
        "circumcision",
        "tattoos/scarification",
        "permanent scarring",
        "extra-ritual in-group markers",
    ]
].sum(axis=1)

# okay try to plot this

# for sorting the plot
wide_sorted = (
    answers_wide.groupby("conflict_type", as_index=False)["composite_score"]
    .mean()
    .sort_values("composite_score")
)

# Get number of observations for the plot
group_counts = answers_wide.groupby("conflict_type").size()

# Create plot
fig, ax = plt.subplots(figsize=(8, 4))
sns.set_style("white")
sns.barplot(
    x="conflict_type",
    y="composite_score",
    data=answers_wide,
    ax=ax,
    order=wide_sorted["conflict_type"],
)
ax.set_xlabel("")
ax.set_xticklabels(
    [f"{label}\n(n={group_counts[label]})" for label in wide_sorted["conflict_type"]],
    fontsize=12,
)
ax.set_ylabel("Mean", fontsize=14)
ax.set_title("")  # ("Extra-ritual in-group markers by conflict type")
