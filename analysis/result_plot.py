import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# load draws
output_files = os.listdir("../data/mdl_output")
draws_files = [i for i in output_files if "draws" in i]
results_files = [i for i in output_files if "results" in i]

# generally useful
convert_labels = {
    "circumcision": "Circumcision",
    "dress": "Dress",
    "extra_ritual_group_markers": "Extra-Ritual In-Group Markers",
    "food_taboos": "Food Taboos",
    "hair": "Hair",
    "ornaments": "Ornaments",
    "permanent_scarring": "Permanent Scarring",
    "tattoos_scarification": "Tattoos/Scarification",
}

# Get the Tab10 colormap
tab10 = plt.get_cmap("tab10")

# Map variables to tab colors using the Tab10 colormap
color_map = {
    list(convert_labels.values())[i]: tab10.colors[i]
    for i in range(len(convert_labels))
}

# Plot 1 (Simplest)
result_list = []
for filename in results_files:
    df = pd.read_csv("../data/mdl_output/" + filename)
    label = filename.split("_results")[0]
    label = convert_labels[label]
    df["Outcome"] = label
    result_list.append(df)
result_df = pd.concat(result_list)

# create position and remove "effect"
result_df["Position"] = result_df["parameter"].map({"intercept": 0, "beta": 1})
result_df = result_df.dropna()

# Plot using Seaborn
plt.figure(figsize=(8, 5))

sns.lineplot(
    data=result_df,
    x="Position",
    y="Estimate",
    hue="Outcome",
    marker="o",
)

plt.savefig("../figures/res_simple_plot.pdf", bbox_inches="tight")
plt.savefig("../figures/png/res_simple_plot.png", bbox_inches="tight", dpi=300)
plt.close()

# Plot 2 (Distributions)
import ptitprince as pt

draws_list = []
for filename in draws_files:
    d = pd.read_csv("../data/mdl_output/" + filename)
    label = filename.split("_draws")[0]
    label = convert_labels[label]
    d["Outcome"] = label
    draws_list.append(d)
df_draws = pd.concat(draws_list)

draws_melted = df_draws.melt(
    id_vars=["Outcome"],
    value_vars=["intercept", "beta", "effect"],
    var_name="Parameter",
    value_name="Value",
)

conversion_dict = {
    "Permanent Scarring": "Permanent\nScarring",
    "Tattoos/Scarification": "Tattoos/\nScarification",
    "Circumcision": "Circumcision",
    "Hair": "Hair",
    "Ornaments": "Ornaments",
    "Food Taboos": "Food Taboos",
    "Dress": "Dress",
    "Extra-Ritual In-Group Markers": "Extra-Ritual In-\nGroup Markers",
}

draws_melted["Parameter"] = draws_melted["Parameter"].map(
    {
        "intercept": "p(Outcome|No External Violent Conflict)",
        "beta": "p(Outcome|External Violent Conflict)",
        "effect": "p(Outcome|E. V. Conflict) - p(Outcome|No E. V. Conflict)",
    }
)

# sorting draws by mean value ;
mean_values = (
    draws_melted[
        draws_melted["Parameter"]
        == "p(Outcome|E. V. Conflict) - p(Outcome|No E. V. Conflict)"
    ]
    .groupby("Outcome")["Value"]
    .mean()
)

# Sort the outcomes based on the calculated means
sorted_outcomes = mean_values.sort_values().index

# Update the order in the DataFrame
draws_melted["Outcome"] = pd.Categorical(
    draws_melted["Outcome"], categories=sorted_outcomes, ordered=True
)

# convert to %
draws_melted["Value"] = draws_melted["Value"] * 100

fig, ax = plt.subplots(figsize=(4, 6), dpi=300)

pt.half_violinplot(
    data=draws_melted,
    y="Outcome",
    x="Value",
    hue="Parameter",
    orient="h",
    split=True,
    alpha=0.65,
    inner=None,
    offset=0,
    scale="area",
    width=1.5,
)
plt.ylabel("")
plt.xlabel("")

# add dotted line
plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)

# Adjust plot to add more space to the top
plt.subplots_adjust(top=0.9)

# Adjust y-axis limits to add more space at the top
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] - 0.5)  # Adjust the value as needed

new_labels = [conversion_dict.get(label, label) for label in sorted_outcomes]
ax.set_yticklabels(new_labels)

# Move the legend to the bottom and remove the title
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.35, -0.05),
    ncol=1,
    frameon=False,
    title="",
)

# Show the plot
plt.savefig("../figures/res_distribution_plot_complete.pdf", bbox_inches="tight")
plt.savefig(
    "../figures/png/res_distribution_plot_complete.png", bbox_inches="tight", dpi=300
)
plt.close()

# Plot 3 (Distributions - Not Difference) #
## NB: THIS PLOT IS NOT FINISHED ##
draws_groups = draws_melted[
    draws_melted["Parameter"]
    != "p(Outcome|E. V. Conflict) - p(Outcome|No E. V. Conflict)"
]

# sorting draws by mean value ;
mean_values = (
    draws_groups[draws_groups["Parameter"] == "p(Outcome|No External Violent Conflict)"]
    .groupby("Outcome")["Value"]
    .mean()
)

# Sort the outcomes based on the calculated means
sorted_outcomes = mean_values.sort_values().index

# Update the order in the DataFrame
draws_groups["Outcome"] = pd.Categorical(
    draws_groups["Outcome"], categories=sorted_outcomes, ordered=True
)

fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
pt.half_violinplot(
    data=draws_groups,
    y="Outcome",
    x="Value",
    hue="Parameter",
    orient="h",
    split=True,
    alpha=0.65,
    inner=None,
    offset=0,
    scale="area",
    width=1.5,
)
plt.ylabel("")
plt.xlabel("")

# Adjust plot to add more space to the top
plt.subplots_adjust(top=0.9)

# Adjust y-axis limits to add more space at the top
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] - 0.5)  # Adjust the value as needed

new_labels = [conversion_dict.get(label, label) for label in sorted_outcomes]
ax.set_yticklabels(new_labels)

# Move the legend to the bottom and remove the title
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.35, -0.05),
    ncol=1,
    frameon=False,
    title="",
)

# Show the plot
plt.savefig("../figures/res_distribution_plot_groups.pdf", bbox_inches="tight")
plt.savefig(
    "../figures/png/res_distribution_plot_groups.png", bbox_inches="tight", dpi=300
)

# Plot 4 (Distributions - Difference)
draws_diff = draws_melted[
    draws_melted["Parameter"]
    == "p(Outcome|E. V. Conflict) - p(Outcome|No E. V. Conflict)"
]
fig, ax = plt.subplots(figsize=(4, 6), dpi=300)

pt.half_violinplot(
    data=draws_diff,
    y="Outcome",
    x="Value",
    hue="Parameter",
    orient="h",
    split=True,
    alpha=0.65,
    inner=None,
    offset=0,
    scale="area",
    width=1.5,
)
plt.ylabel("")
plt.xlabel("")
plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
