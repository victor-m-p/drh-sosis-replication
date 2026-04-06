"""
vmp 2026-03-29
Merge answerset with language tip mapping and save per-marker CSVs for two analyses:

  model/external/input/          predictor: violent_external (all mapped entries)
  model/internal/input/          predictor: violent_internal (violent_external == 0)
  model/external_noeHRAF/input/  same as external, excluding eHRAF entries
  model/internal_noeHRAF/input/  same as internal, excluding eHRAF entries

Input:  data/preprocessed/answerset.csv
        data/preprocessed/tip_map.csv
Output: data/model/{analysis}/input/{marker}.csv
"""

import os
import pandas as pd
from helper_functions import process_time_region

answerset = pd.read_csv("../data/preprocessed/answerset.csv")
tip_map   = pd.read_csv("../data/preprocessed/tip_map.csv")[
    ["entry_id", "tip_name", "ID", "Glottocode"]
]

# merge: inner join drops entries without a tree tip
answerset_phylo = answerset.merge(tip_map, on="entry_id", how="inner")
print(f"Entries after tip mapping: {answerset_phylo['entry_id'].nunique()}")

dependent_variables = [
    "circumcision", "tattoos_scarification", "permanent_scarring",
    "extra_ritual_group_markers", "food_taboos", "hair", "dress", "ornaments",
]

# ── 1. External warfare analysis (full sample) ─────────────────────────────────

os.makedirs("../data/model/external/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_phylo, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/external/input/{dv}.csv", index=False)
    print(f"external / {dv}: {len(out)} rows, {out['tip_name'].nunique()} unique tips")

# ── 2. Internal warfare analysis (violent_external == 0 only) ──────────────────

answerset_no_ext = answerset_phylo[answerset_phylo["violent_external"] == 0]
print(f"\nEntries with violent_external == 0: {answerset_no_ext['entry_id'].nunique()}")

os.makedirs("../data/model/internal/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ext, "entry_id", "violent_internal", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal/input/{dv}.csv", index=False)
    print(f"internal / {dv}: {len(out)} rows, {out['tip_name'].nunique()} unique tips")

# ── 3. External warfare analysis — eHRAF excluded ─────────────────────────────

answerset_no_ehraf = answerset_phylo[answerset_phylo["data_source"] != "eHRAF"]
print(f"\nEntries excluding eHRAF: {answerset_no_ehraf['entry_id'].nunique()}")

os.makedirs("../data/model/external_noeHRAF/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ehraf, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/external_noeHRAF/input/{dv}.csv", index=False)
    print(f"external_noeHRAF / {dv}: {len(out)} rows, {out['tip_name'].nunique()} unique tips")

# ── 4. Internal warfare analysis — eHRAF excluded ─────────────────────────────

answerset_no_ext_no_ehraf = answerset_no_ext[answerset_no_ext["data_source"] != "eHRAF"]
print(f"\nEntries with violent_external == 0, excluding eHRAF: {answerset_no_ext_no_ehraf['entry_id'].nunique()}")

os.makedirs("../data/model/internal_noeHRAF/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ext_no_ehraf, "entry_id", "violent_internal", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal_noeHRAF/input/{dv}.csv", index=False)
    print(f"internal_noeHRAF / {dv}: {len(out)} rows, {out['tip_name'].nunique()} unique tips")
