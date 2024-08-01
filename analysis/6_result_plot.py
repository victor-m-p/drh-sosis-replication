""" 
VMP 2024-07-31
This script generates the plot for the results of the Bayesian analysis (Figure 2).
We are using draws from the Bayesian analysis (brms_models.R) here.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# load draws
output_files = os.listdir("../data/mdl_output")
draws_files = [i for i in output_files if "draws" in i]

# generally useful
convert_labels = {
    "circumcision": "Circumcision",
    "dress": "Dress",
    "extra_ritual_group_markers": "Extra-Ritual In-Group Markers",
    "food_taboos": "Food Taboos",
    "hair": "Hair",
    "ornaments": "Ornaments",
    "permanent_scarring": "Permanent Scarring",
    "tattoos_scarification": "Tattoos or Scarification",
}

# Plot 2 (Distributions)
import ptitprince as pt

draws_list = []
for filename in draws_files:
    d = pd.read_csv("../data/mdl_output/" + filename)
    label = filename.split("_draws")[0]
    label = convert_labels[label]
    d["Marker"] = label
    draws_list.append(d)
df_draws = pd.concat(draws_list)

draws_melted = df_draws.melt(
    id_vars=["Marker"],
    value_vars=["intercept", "beta", "effect"],
    var_name="Parameter",
    value_name="Value",
)

conversion_dict = {
    "Permanent Scarring": "Permanent\nScarring",
    "Tattoos or Scarification": "Tattoos or\nScarification",
    "Circumcision": "Circumcision",
    "Hair": "Hair",
    "Ornaments": "Ornaments",
    "Food Taboos": "Food Taboos",
    "Dress": "Dress",
    "Extra-Ritual In-Group Markers": "Extra-Ritual In-\nGroup Markers",
}

draws_melted["Parameter"] = draws_melted["Parameter"].map(
    {
        "intercept": "p(Marker|No External Violent Conflict)",
        "beta": "p(Marker|External Violent Conflict)",
        "effect": "p(Marker|E. V. Conflict) - p(Marker|No E. V. Conflict)",
    }
)

# sorting draws by mean value ;
mean_values = (
    draws_melted[
        draws_melted["Parameter"]
        == "p(Marker|E. V. Conflict) - p(Marker|No E. V. Conflict)"
    ]
    .groupby("Marker")["Value"]
    .mean()
)

# Sort the Markers based on the calculated means
sorted_Markers = mean_values.sort_values().index

# Update the order in the DataFrame
draws_melted["Marker"] = pd.Categorical(
    draws_melted["Marker"], categories=sorted_Markers, ordered=True
)

# convert to %
draws_melted["Value"] = draws_melted["Value"] * 100

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

pt.half_violinplot(
    data=draws_melted,
    y="Marker",
    x="Value",
    hue="Parameter",
    orient="h",
    split=True,
    alpha=0.65,
    inner=None,
    offset=-0.2,
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
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] - 0.3)  # Adjust the value as needed

new_labels = [conversion_dict.get(label, label) for label in sorted_Markers]
ax.set_yticklabels(new_labels)

# Move the legend to the bottom and remove the title
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.32, -0.05),
    ncol=1,
    frameon=False,
    title="",
)

# Show the plot
plt.savefig("../figures/bayesian_figure.pdf", bbox_inches="tight")
plt.savefig("../figures/png/bayesian_figure.png", bbox_inches="tight", dpi=300)
plt.close()
