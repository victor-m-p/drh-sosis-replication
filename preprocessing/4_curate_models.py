"""
vmp 2026-03-29
Merge answerset with language tip mapping and save per-marker CSVs for these analyses:

  model/external/input/             predictor: violent_external (all mapped entries)
  model/internal/input/             predictor: violent_internal (violent_external == 0)
  model/external_noeHRAF/input/     same as external, excluding eHRAF entries
  model/internal_noeHRAF/input/     same as internal, excluding eHRAF entries
  model/internal_only/input/        predictor: internal_only (Sosis et al. 2007's grouping —
                                     internal conflict present AND external conflict absent,
                                     vs. everything else; see internal_only_flag below)
  model/internal_only_noeHRAF/input/  same as internal_only, excluding eHRAF entries

Input:  data/preprocessed/answerset.csv
        data/preprocessed/language_master.csv
Output: data/model/{analysis}/input/{marker}.csv
        data/preprocessed/tip_map.csv (entry_id -> tip_name, for 6_language_phylo* and 7_covariance_matrix.Rmd)
"""

import os
import pandas as pd
from helper_functions import process_time_region

answerset = pd.read_csv("../data/preprocessed/answerset.csv")

language_master = pd.read_csv("../data/preprocessed/language_master.csv")
tip_map = language_master[["entry_id", "tip_assigned"]].rename(columns={"tip_assigned": "tip_name"})
tip_map.to_csv("../data/preprocessed/tip_map.csv", index=False)
dependent_variables = [
    "circumcision", "tattoos_scarification", "permanent_scarring",
    "extra_ritual_group_markers", "food_taboos", "hair", "dress", "ornaments",
]

# internal_only: 1 if internal conflict present and external conflict absent, else 0.
# NaN if internal itself is missing (excluded downstream). Computed on the full answerset
# before any filtering, so answerset_no_ehraf (below) inherits it directly.
def internal_only_flag(vi, ve):
    if pd.isna(vi):
        return float("nan")
    return int(vi == 1 and ve == 0)

answerset["internal_only"] = answerset.apply(
    lambda r: internal_only_flag(r["violent_internal"], r["violent_external"]), axis=1
)

# 1. External warfare analysis (full sample)
os.makedirs("../data/model/external/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/external/input/{dv}.csv", index=False)

# 2. Internal warfare analysis (violent_external == 0 only)
answerset_no_ext = answerset[answerset["violent_external"] == 0]
os.makedirs("../data/model/internal/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ext, "entry_id", "violent_internal", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal/input/{dv}.csv", index=False)

# 3. External warfare analysis — eHRAF excluded
answerset_no_ehraf = answerset[answerset["data_source"] != "eHRAF"]

os.makedirs("../data/model/external_noeHRAF/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ehraf, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/external_noeHRAF/input/{dv}.csv", index=False)

# 4. Internal warfare analysis — eHRAF excluded
answerset_no_ext_no_ehraf = answerset_no_ext[answerset_no_ext["data_source"] != "eHRAF"]

os.makedirs("../data/model/internal_noeHRAF/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ext_no_ehraf, "entry_id", "violent_internal", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal_noeHRAF/input/{dv}.csv", index=False)

# 5. Internal-only warfare analysis (full sample)
os.makedirs("../data/model/internal_only/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset, "entry_id", "internal_only", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal_only/input/{dv}.csv", index=False)

# 6. Internal-only warfare analysis — eHRAF excluded
os.makedirs("../data/model/internal_only_noeHRAF/input", exist_ok=True)

for dv in dependent_variables:
    out = process_time_region(
        answerset_no_ehraf, "entry_id", "internal_only", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/internal_only_noeHRAF/input/{dv}.csv", index=False)

# 7. External warfare analysis, restricted to entries where markers are present
#
# The six markers below are child questions of "Are extra-ritual in-group markers
# present"; a coder only sees them when that parent is answered "Yes". 1_curate_data.py
# therefore infers "No" for every child of a "No" parent (see fill_answers in
# helper_functions.py), which makes roughly 62% of each child marker's rows a copy of
# the parent's answer rather than an observed coding. Restricting to parent == 1 drops
# exactly those inferred rows, so these models ask "given that markers are present,
# does external conflict predict which kind?" rather than "are markers present at all?".
child_variables = [
    "circumcision", "tattoos_scarification",       # permanent
    "dress", "food_taboos", "hair", "ornaments",   # transitory
]

# The filter above is only equivalent to "drop the inferred answers" as long as no child
# answer was ever observed under a "No" or missing parent. That holds in SCCSR.v3; assert
# it so a later data release cannot silently invalidate the restriction.
parent_no = answerset[answerset["extra_ritual_group_markers"] == 0]
parent_missing = answerset[answerset["extra_ritual_group_markers"].isna()]
assert (parent_no[child_variables].fillna(0) == 0).all().all(), \
    "observed 'Yes' child answer under a 'No' parent: parent filter would drop observed data"
assert parent_missing[child_variables].isna().all().all(), \
    "observed child answer under a missing parent: parent filter would drop observed data"

answerset_markers_present = answerset[answerset["extra_ritual_group_markers"] == 1]

os.makedirs("../data/model/external_markers_present/input", exist_ok=True)

for dv in child_variables:
    out = process_time_region(
        answerset_markers_present, "entry_id", "violent_external", dv, "year_scaled", "world_region",
    )
    out = out.merge(tip_map, on="entry_id", how="left")
    out.to_csv(f"../data/model/external_markers_present/input/{dv}.csv", index=False)
