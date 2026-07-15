'''
Checking out some institutional variables.
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

columns = [
    # the two main predictors
    'violent_external',
    'violent_internal',
    # institutions (focus on other)
    'judges_other',
    #'judges_own',
    'legal_code_other',
    #'legal_code_own',
    'military_participate',
    #'military_possess',
    'military_protected',
    'police_force_other',
    #'police_force_own',
    'punish_other',
    #'punish_own',
    # the two main markers (all others are sub)
    'extra_ritual_group_markers',
    'permanent_scarring',
    'state',
]

df = pd.read_csv("../data/preprocessed/answerset_large.csv")

corr = df[columns].corr()

# Reorder by hierarchical clustering
dist = ((1 - corr) + (1 - corr).T) / 2
dist = dist.clip(lower=0)
link = linkage(squareform(dist.values, checks=False), method='average')
order = leaves_list(link)
ordered_cols = [columns[i] for i in order]
corr_ordered = corr.loc[ordered_cols, ordered_cols]

# Compute present−absent year differences
diffs = []
for col in ordered_cols:
    mean_present = df.loc[df[col] == 1, 'year_from'].mean()
    mean_absent  = df.loc[df[col] == 0, 'year_from'].mean()
    diffs.append(mean_present - mean_absent)

fig, (ax_heat, ax_bar) = plt.subplots(
    1, 2,
    figsize=(13, 8),
    gridspec_kw={'width_ratios': [5, 1], 'wspace': 0.05}
)

sns.heatmap(
    corr_ordered,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax_heat
)

# Highlight warfare rows
n = len(ordered_cols)
for col in ('violent_external', 'violent_internal'):
    row_idx = ordered_cols.index(col)
    ax_heat.add_patch(mpatches.Rectangle(
        (0, row_idx), n, 1,
        fill=False, edgecolor='black', lw=2.5, clip_on=False
    ))

ax_heat.set_title("Pairwise correlations (clustered)")

# Bar chart: present minus absent, rows aligned with heatmap
y = range(len(ordered_cols))
colors = ['#d73027' if d > 0 else '#4575b4' for d in diffs]
ax_bar.barh(list(y), diffs, color=colors, height=0.6, align='center')
ax_bar.axvline(0, color='black', linewidth=0.8)
ax_bar.invert_yaxis()
ax_bar.set_yticks([])
ax_bar.set_xlabel("present − absent\n(mean year CE)", fontsize=9)
ax_bar.tick_params(axis='x', labelsize=8)
ax_bar.spines[['top', 'right', 'left']].set_visible(False)

plt.savefig("../figures/EDA_correlation/heatmap_corr.png", dpi=150, bbox_inches='tight')
plt.show()