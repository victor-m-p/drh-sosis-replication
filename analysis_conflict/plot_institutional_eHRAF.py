"""
Type of violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# find entries that are eHRAF or Pulotu
entry_data = pd.read_csv("../data/raw/entry_data.csv")
entry_data = entry_data[["entry_id", "data_source"]].drop_duplicates()
entry_data = entry_data[
    (entry_data["data_source"] == "eHRAF")
    | (entry_data["data_source"].str.contains("Pulotu"))
]

# inner join with answer data
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")
answers = answers.merge(entry_data, on="entry_id", how="inner")

# answers to wide format
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# select columns
select_columns = [
    # IV
    "violent external",
    "violent internal",
    # DV
    "group judges",
    "group legal code",
    "group police force",
    "group punishment",
    "other judicial system",
    "other legal code",
    "other police force",
    "other punishment",
]
answers_wide = answers_wide[select_columns]

# remove nan in IV
answers_wide = answers_wide.dropna(subset=["violent external", "violent internal"])

# code conflict
from helper_functions import code_conflict

answers_wide["conflict_type"] = answers_wide.apply(code_conflict, axis=1)

dv = answers_wide.drop(
    columns=["violent external", "violent internal", "conflict_type"]
).columns.tolist()

order = [
    "No violent conflict",
    "Internal only",
    "Internal and external",
    "External only",
]
answers_wide["conflict_type"] = pd.Categorical(
    answers_wide["conflict_type"], categories=order, ordered=True
)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 7))
sns.set_style("white")
for i, dependent_variable in enumerate(dv):
    bplot = sns.barplot(
        x="conflict_type",
        y=dependent_variable,
        data=answers_wide,
        ax=ax[i // 4, i % 4],
        order=order,  # Ensure the bars are in the desired order
    )
    ax[i // 4, i % 4].set_xlabel("")  # Clear x-axis labels
    ax[i // 4, i % 4].set_xticklabels([])  # Clear x-tick labels
    ax[i // 4, i % 4].set_ylabel(
        "Fraction Yes" if i % 4 == 0 else "", fontsize=14
    )  # Set y-label only for the first column
    ax[i // 4, i % 4].set_title(f"{dependent_variable}", fontsize=16)

# Creating a manual legend for the x-axis categories
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=bplot.patches[i].get_facecolor(), label=order[i])
    for i in range(len(order))
]
fig.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
    title="Conflict Type",
    title_fontsize=16,
    fontsize=16,
    frameon=False,
)

plt.tight_layout()
plt.savefig("../figures/institutional_eHRAF_pulotu.pdf", bbox_inches="tight")
plt.savefig(
    "../figures/png/institutional_eHRAF_pulotu.png", bbox_inches="tight", dpi=300
)
