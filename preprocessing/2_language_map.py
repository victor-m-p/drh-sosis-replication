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

# ── 1. Entries we need to map ──────────────────────────────────────────────────
answerset = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = set(answerset["entry_id"].unique())

# ── 2. Load DRH language tags ──────────────────────────────────────────────────
drh_tags  = pd.read_csv("../data/raw/entity_tags.csv")
drh_tags  = drh_tags[drh_tags["entry_id"].isin(entry_ids)]

drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]
drh_langs = drh_langs.drop_duplicates()
drh_langs = drh_langs.dropna(subset="entrytag_name")
drh_langs["coding"] = "SCCSR.v3"

# --- 2.1. entries with no language tag, coded by Matthew Hamm ----------------
manual_codes_A = pd.read_csv("../data/raw/manual_lang_A.csv")
drh_langs = pd.concat([drh_langs, manual_codes_A], ignore_index=True)
drh_langs = drh_langs.drop_duplicates()

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

drh_tags[drh_tags['entry_id']==1013]
