import pandas as pd
import numpy as np
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

### external violent conflict ###
# make a first plot with permanent scarring and extra-ritual in-group markers against warfare.
answers_wide = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)

# take out relevant columns for now
wide_subset = answers_wide[
    [
        "extra-ritual in-group markers",
        "circumcision",
        "tattoos/scarification",
        "permanent scarring",
        "violent external",
    ]
]

wide_subset.isna().sum()  # explains why this is infeasible.

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

markers = [
    "extra-ritual in-group markers",
    "circumcision",
    "tattoos/scarification",
    "permanent scarring",
]
chi2_results = {marker: run_chi2_test(df_long, marker) for marker in markers}

# Prepare labels
labels = [
    (
        f"{marker}\n(χ²={result[0]:.2f}; p<0.05)"
        if result[1] < 0.05
        else f"{marker}\n(χ²={result[0]:.2f}; ns)"
    )
    for marker, result in chi2_results.items()
]

# Fix labels
labels[0] = labels[0].replace(
    "extra-ritual in-group markers", "extra-ritual \nin-group markers"
)
labels[2] = labels[2].replace("tattoos/scarification", "tattoos/\nscarification")
labels[3] = labels[3].replace("permanent scarring", "permanent\nscarring")

# Assuming the data transformation is done
plt.figure(figsize=(9, 4))
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
plt.title("")
plt.ylabel("Fraction Yes", fontsize=14)
plt.xlabel("")
plt.legend(title="", loc="upper right")

### clearly unfeasible for sub-questions ###

### try with super-questions ###
