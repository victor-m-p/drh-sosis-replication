'''
Multiple Correspondence Analysis (MCA) on the substantive DRH variables (markers,
conflict, enforcement/warfare institutions, state), entry metadata excluded.
Missingness is kept as its own explicit category per variable, rather than dropped,
so entries aren't discarded just because one of ~20 variables is missing.
'''

import pandas as pd
import matplotlib.pyplot as plt
import prince

columns = [
    # markers
    #'circumcision', 
    #'dress', 
    'extra_ritual_group_markers', 
    #'food_taboos', 
    #'hair',
    #'ornaments',
    'permanent_scarring', 
    #'tattoos_scarification',
    # conflict
    'violent_external', 
    'violent_internal',
    # institutions
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
    # societal scale -- 'state' only, 'society_type' left out since it's the same
    # information at finer resolution and would just show up as a near-duplicate
    'state',
]

family = {}
for c in ['circumcision', 'dress', 'extra_ritual_group_markers', 'food_taboos', 'hair',
          'ornaments', 'permanent_scarring', 'tattoos_scarification']:
    family[c] = 'marker'
for c in ['violent_external', 'violent_internal']:
    family[c] = 'conflict'
for c in ['judges_other', 'judges_own', 'legal_code_other', 'legal_code_own',
          'military_participate', 'military_possess', 'military_protected',
          'police_force_other', 'police_force_own', 'punish_other', 'punish_own']:
    family[c] = 'institution'
family['state'] = 'state'

family_colors = {
    'marker': '#4C72B0',
    'conflict': '#DD8452',
    'institution': '#55A868',
    'state': '#C44E52',
}

df = pd.read_csv("../data/preprocessed/answerset_large.csv")
X = df[columns].copy()

# recode 0/1 -> Yes/No and NaN -> "Missing", so missingness is its own category
# instead of causing row deletion
for c in columns:
    X[c] = X[c].map({0: "No", 1: "Yes"}).fillna("Missing")
X = X.astype("category")

mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(X)

print(mca.eigenvalues_summary)

row_coords = mca.row_coordinates(X)
col_coords = mca.column_coordinates(X)

fig, ax = plt.subplots(figsize=(11, 9))

# entries in the background, small and light
ax.scatter(row_coords[0], row_coords[1], color='lightgray', s=10, alpha=0.4, zorder=1)

# variable-level points, colored by family, direct-labeled
for idx, row in col_coords.iterrows():
    var, level = idx.rsplit('__', 1)
    color = family_colors[family[var]]
    ax.scatter(row[0], row[1], color=color, s=40, zorder=2)
    ax.annotate(f"{var}={level}", (row[0], row[1]), fontsize=7, color=color,
                xytext=(3, 3), textcoords='offset points')

ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel(f"Dim 1 ({mca.percentage_of_variance_[0]:.1f}%)")
ax.set_ylabel(f"Dim 2 ({mca.percentage_of_variance_[1]:.1f}%)")

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=f, markersize=8)
           for f, c in family_colors.items()]
ax.legend(handles=handles, loc='best')

plt.tight_layout()
plt.savefig("../figures/EDA_correlation/mca_all_variables.png", dpi=150, bbox_inches='tight')
plt.show()
