"""
vmp 2026-03-29
Map Religious Group entries to tips in the ASJP language phylogeny (Jäger 2018).

Pipeline:
  DRH entity tags -> Glottolog (name -> Glottocode) -> ASJP (Glottocode -> ID) -> world.tre (ID -> tip_name)

Input:  data/preprocessed/answerset.csv       (551 entries)
Output: data/preprocessed/tip_map.csv         (entry_id -> tip_name, one row per entry)
"""

import pandas as pd
from Bio import Phylo

# 1. Entries we need to map 
answerset = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = set(answerset["entry_id"].unique())

# 2. Load DRH language tags
drh_tags  = pd.read_csv("../data/raw/entity_tags.csv")

''' Key observations:
- We do not have Glottocode which would be cleanest.
- Entrytag level increases with specificity. 
- Each Entry has multiple rows which is a nested lineage.
'''

# only worry about relevant entries (those in the answerset)
drh_tags  = drh_tags[drh_tags["entry_id"].isin(entry_ids)]

# remove tags not related to language and select relevant columns
drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]
drh_langs = drh_langs.drop_duplicates()
drh_langs = drh_langs.dropna(subset="entrytag_name")

# save coding as SCCSR.v3 when entrytag from DRH chain (not manual)
drh_langs["coding"] = "SCCSR.v3" 

# --- 2.1. entries with no language tag, coded by Matthew Hamm ----------------
manual_codes_A = pd.read_csv("../data/raw/manual_lang_A.csv")
drh_langs = pd.concat([drh_langs, manual_codes_A], ignore_index=True)
drh_langs = drh_langs.drop_duplicates()
manual_codes_A[~manual_codes_A.entry_id.isin(entry_ids)]

# --- 2.2. entries whose tags could not be resolved to a tree tip --------------
# several experts assigned tip_name directly (entrytag_level/path stay NaN)
manual_codes_B = pd.read_csv("../data/raw/manual_lang_B.csv")

# ── 3. Load reference tables ───────────────────────────────────────────────────
glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["Glottocode", "Glottolog_Name"]

asjp      = pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Glottocode"]].drop_duplicates()
tree      = Phylo.read("../data/jaeger2018/world.tre", "newick")
tips_by_id = {t.name.split(".")[-1]: t.name for t in tree.get_terminals()}

# ── 4. Chain lookups on unique tag names ───────────────────────────────────────
tag_names = drh_langs[["entrytag_name"]].drop_duplicates()

tip_map = (
    tag_names
    .merge(glottolog, left_on="entrytag_name", right_on="Glottolog_Name", how="left")
    .merge(asjp, on="Glottocode", how="left")
    .assign(tip_name=lambda d: d["ID"].map(tips_by_id))
    [["entrytag_name", "Glottocode", "ID", "tip_name"]]
)

drh_mapped = drh_langs.merge(tip_map, on="entrytag_name", how="left")

# ── 4.1. Apply manual tip overrides (manual_codes_B) ──────────────────────────
rows_B     = manual_codes_B[["entry_id", "entrytag_name", "Glottocode", "tip_name"]].copy()
drh_mapped = pd.concat([drh_mapped, rows_B], ignore_index=True)

# ── 5. Final mapping: one tip per entry (deepest matched level) ────────────────
entry_tip = (
    drh_mapped[drh_mapped["tip_name"].notna()]
    .sort_values("entrytag_level", ascending=False)
    .drop_duplicates("entry_id")
    [["entry_id", "entrytag_name", "entrytag_path", "Glottocode", "ID", "tip_name", "coding"]]
    .sort_values("entry_id")
    .reset_index(drop=True)
)

# ── 6. Save ────────────────────────────────────────────────────────────────────
entry_tip.to_csv("../data/preprocessed/tip_map.csv", index=False)

### more checks ###
deepest = (drh_langs
           .sort_values("entrytag_level", ascending=False)
           .drop_duplicates("entry_id")
           [["entry_id", "entrytag_name", "entrytag_level"]])

manual_A = pd.read_csv("../data/raw/manual_lang_A.csv")
manual_B = pd.read_csv("../data/raw/manual_lang_B.csv")

auto_entries   = set(deepest["entry_id"])           # entries with a DRH language chain
manual_entries = set(manual_A["entry_id"]) | set(manual_B["entry_id"])

print("entries via DRH chain:", len(auto_entries))
print("entries via manual  :", len(manual_entries))
print("A/B overlap with each other:", set(manual_A['entry_id']) & set(manual_B['entry_id']))
print("manual entries that ALSO have a DRH chain (potential collision):",
      len(auto_entries & manual_entries))
print(sorted(auto_entries & manual_entries)[:20])

# and of those collisions, how deep is the DRH tag the auto-branch would pick?
collision = deepest[deepest["entry_id"].isin(auto_entries & manual_entries)]
print("\nDeepest-level distribution for colliding entries:")
print(collision["entrytag_level"].value_counts().sort_index())

### more checks ###
drh_only = drh_langs[drh_langs["coding"] == "SCCSR.v3"]   # DRH tags only, before manual A
deepest = (drh_only
           .sort_values("entrytag_level", ascending=False)
           .drop_duplicates("entry_id")
           [["entry_id", "entrytag_name", "entrytag_level"]])

print("=== manual_A columns/shape ===", manual_A.shape); print(manual_A.head(10).to_string())
print("\n=== manual_B columns/shape ===", manual_B.shape); print(manual_B.head(10).to_string())

# an entry that's in BOTH A and B and also has a deep DRH chain — the maximal collision
eid = 173
print(f"\n=== entry {eid}: DRH chain ===")
print(drh_only[drh_only.entry_id==eid][["entrytag_name","entrytag_level"]].sort_values("entrytag_level").to_string())
print(f"\n=== entry {eid}: in manual_A ===");  print(manual_A[manual_A.entry_id==eid].to_string())
print(f"\n=== entry {eid}: in manual_B ===");  print(manual_B[manual_B.entry_id==eid].to_string())



# ...
# entries whose ONLY path to a tip is manual (DRH tag is family-level / won't resolve)
drh_deepest = (drh_only.sort_values("entrytag_level", ascending=False)
                       .drop_duplicates("entry_id")[["entry_id","entrytag_name","entrytag_level"]])

manual_all = set(manual_A.entry_id) | set(manual_B.entry_id)

# of the 84 manual entries, how deep is their DRH tag? shallow => manual is doing real rescue work
probe = drh_deepest[drh_deepest.entry_id.isin(manual_all)]
print("manual entries whose DRH deepest tag is FAMILY-level (<=3):",
      (probe.entrytag_level <= 3).sum())
print("manual entries whose DRH deepest tag is deep (>=7):",
      (probe.entrytag_level >= 7).sum())
print("\nfull level distribution of DRH tag for the 84 manual entries:")
print(probe.entrytag_level.value_counts().sort_index())
print("\nentries in manual set with NO DRH chain at all:",
      len(manual_all - set(drh_only.entry_id)))




#

victims = [607, 1820, 1829, 1835, 1938, 941, 1522, 1985, 1993, 2069, 2114, 2363, 2420]

# what names do these entries carry, and what does the name→glottocode→ID chain produce?
probe = (drh_langs[drh_langs.entry_id.isin(victims)][["entry_id","entrytag_name","coding"]]
         .merge(tip_map, on="entrytag_name", how="left"))
print(probe.sort_values("entry_id").to_string())

# the crux: what is tips_by_id doing with a NaN (or missing) key?
import numpy as np
print("\nNaN in tips_by_id keys? ->", any(pd.isna(k) for k in tips_by_id))
print("tips_by_id.get(np.nan) ->", tips_by_id.get(np.nan, "· MISSING ·"))
print("tips_by_id maps ALLENTIAC-tip from which key? ->",
      [k for k,v in tips_by_id.items() if v=="Hur.HUARPE.ALLENTIAC"])


# how many tag names fail the Glottolog match -> NaN Glottocode -> will blow up on asjp merge?
bad = tag_names.merge(glottolog, left_on="entrytag_name", right_on="Glottolog_Name", how="left")
n_nan_glotto = bad["Glottocode"].isna().sum()
print("tag names with NO Glottolog match (NaN Glottocode):", n_nan_glotto, "of", len(bad))
print(sorted(bad.loc[bad["Glottocode"].isna(), "entrytag_name"].unique())[:40])

# and how many ASJP rows have NaN Glottocode (the size of the blowup set)?
print("\nASJP rows with NaN Glottocode:", asjp["Glottocode"].isna().sum(), "of", len(asjp))