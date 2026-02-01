""" 
VMP 2026-01-31 (updated.)
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

from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

param_order = [
    "p(Marker|No External Violent Conflict)",
    "p(Marker|External Violent Conflict)",
    "p(Marker|E. V. Conflict) - p(Marker|No E. V. Conflict)",
]

# pick colors (match seaborn defaults you had: blue/orange/green)
colors = sns.color_palette("tab10", 3)
param_color = {
    param_order[0]: colors[0],
    param_order[1]: colors[1],
    param_order[2]: colors[2],
}

# draw each distribution as its own half-violin (no hue, no split)
for p in param_order:
    pt.half_violinplot(
        data=draws_melted[draws_melted["Parameter"] == p],
        y="Marker",
        x="Value",
        orient="h",
        color=param_color[p],
        alpha=0.7,
        inner=None,
        offset=-0.2,   
        scale="width", #"area",
        width=1.5,
        ax=ax,
    )

plt.ylabel("")
plt.xlabel("")
plt.axvline(x=0, color="black", linestyle="--", linewidth=0.5)

plt.subplots_adjust(top=0.9)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] - 0.3)

new_labels = [conversion_dict.get(label, label) for label in sorted_Markers]
ax.set_yticklabels(new_labels)

# manual legend (deterministic)
legend_handles = [Patch(facecolor=param_color[p], label=p) for p in param_order]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.32, -0.05),
    ncol=1,
    frameon=False,
    title="",
)

plt.savefig("../figures/bayesian_figure.pdf", bbox_inches="tight")
plt.savefig("../figures/png/bayesian_figure.png", bbox_inches="tight", dpi=300)
plt.close()