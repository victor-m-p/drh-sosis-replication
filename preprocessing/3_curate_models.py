"""
vmp 2026-03-29
Merge answerset with language tip mapping and save per-marker CSVs to phylo_input/.

Input:  data/preprocessed/answerset.csv   (551 entries)
        data/preprocessed/tip_map.csv     (457 entries mapped to tree tips)
Output: data/phylo_input/{marker}.csv     (one file per dependent variable)
"""

import os
import pandas as pd
from helper_functions import process_time_region

answerset = pd.read_csv("../data/preprocessed/answerset.csv")
tip_map   = pd.read_csv("../data/preprocessed/tip_map.csv")[
    ["entry_id", "tip_name", "ID", "Glottocode"]
]

# merge: inner join drops the 94 entries without a tree tip
answerset_phylo = answerset.merge(tip_map, on="entry_id", how="inner")
print(f"Entries after tip mapping: {answerset_phylo['entry_id'].nunique()}")  # 457

dependent_variables = [
    "circumcision", "tattoos_scarification", "permanent_scarring",
    "extra_ritual_group_markers", "food_taboos", "hair", "dress", "ornaments",
]

os.makedirs("../data/phylo_input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_phylo, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    # re-attach tip columns dropped by process_time_region
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/phylo_input/{dv}.csv", index=False)
    print(f"{dv}: {len(out)} rows, {out['tip_name'].nunique()} unique tips")
