import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

n_regions = 10

convert_labels = {
    "archaic_language": "Archaic Language",
    "circumcision": "Circumcision",
    "dress": "Dress",
    "extra_ritual_markers": "Extra-Ritual In-Group Markers",
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

summary_files_pct = os.listdir("../mdl_output/pct")
summary_files_pct = sorted(summary_files_pct)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
handles = []
for filename in summary_files_pct:
    d = pd.read_csv("../mdl_output/pct/" + filename)

    # extract main effects
    intercept = d[d["parameter"] == "Intercept"]["mean_pct"].values[0]
    intercept_slope = d[d["parameter"] == "Intercept + Slope"]["mean_pct"].values[0]

    # extract random effects
    random_intercepts = [
        d[d["parameter"] == f"intercept_{i}"]["mean_pct"].values[0]
        for i in range(n_regions)
    ]
    random_intercepts_slopes = [
        d[d["parameter"] == f"intercept + slope_{i}"]["mean_pct"].values[0]
        for i in range(n_regions)
    ]
    random_effects = np.array([random_intercepts, random_intercepts_slopes]).T
    random_effects = pd.DataFrame(
        random_effects, columns=["intercept", "intercept + slope"]
    )

    # extract variable name
    label = filename.split("_summary")[0]
    label = convert_labels[label]

    # get color
    color = color_map[label]

    (main_effect_plot,) = ax.plot(
        ["No External Violent Conflict", "External Violent Conflict"],
        [intercept, intercept_slope],
        color=color,
        label=label,
        marker="o",
    )

    handles.append(main_effect_plot)  # Append to handles for legend

    # Plot random effects
    for x, y in zip(random_intercepts, random_intercepts_slopes):
        ax.plot(
            ["No External Violent Conflict", "External Violent Conflict"],
            [x, y],
            color=color,
            alpha=0.1,
            marker="o",
        )

# Adjust x-axis and label sizes
ax.set_xticklabels(
    ["No External\nViolent Conflict", "External\nViolent Conflict"], fontsize=14
)

# Place the legend outside of the figure
ax.legend(
    handles=handles,
    title="Dependent Variable",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=12,
    title_fontsize=14,
    frameon=False,
)

ax.set_facecolor("white")

plt.ylabel("Estimate Percent Yes", fontsize=14)

plt.tight_layout()  # Adjust the layout to make room for the legend
plt.savefig("../figures/pymc_plot_pct.pdf", bbox_inches="tight")
plt.savefig("../figures/png/pymc_plot_pct.png", bbox_inches="tight", dpi=300)
