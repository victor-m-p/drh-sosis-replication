'''
Build phylo_input files for the phylogenetic analysis.

For each file in /data/mdl_input/:
  - merge with the language tip mapping (entry_id -> tip_name)
  - drop entries that cannot be mapped to the tree
  - save to /data/phylo_input/

Tip mapping pipeline:
  DRH entity tags -> Glottolog (name -> Glottocode) -> ASJP (Glottocode -> ID) -> world.tre (ID -> tip_name)
'''

import glob
import os
import pandas as pd
from Bio import Phylo

# ── 1. Identify relevant entries (those that appear in any mdl_input file) ──

mdl_files = sorted(glob.glob("../data/mdl_input/*.csv"))
mdl_entries = set()
for f in mdl_files:
    mdl_entries.update(pd.read_csv(f)["entry_id"].unique())

print(f"Unique entries across all mdl_input files: {len(mdl_entries)}")

# ── 2. Load DRH language tags, restricted to relevant entries ──

drh_tags = pd.read_csv("../data/raw/entity_tags.csv")
drh_tags = drh_tags[drh_tags["entry_id"].isin(mdl_entries)]

drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]

print(f"Entries with a language tag: {drh_langs['entry_id'].nunique()}")

# ── 3. Build tip mapping: entrytag_name -> Glottocode -> ASJP ID -> tip_name ──

# Glottolog: name -> Glottocode
glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["Glottocode", "Glottolog_Name"]

# ASJP: Glottocode -> ID
asjp = pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Glottocode"]].drop_duplicates()

# Tree: ID -> full tip label
tree = Phylo.read("../asjp/raw/world.tre", "newick")
tips = [t.name for t in tree.get_terminals()]
tips_by_id = {t.split(".")[-1]: t for t in tips}

# Chain the lookups on unique tag names (avoids redundant work)
tag_names = drh_langs[["entrytag_name"]].drop_duplicates()

tip_map = (tag_names
    .merge(glottolog, left_on="entrytag_name", right_on="Glottolog_Name", how="left")
    .merge(asjp, on="Glottocode", how="left")
    .assign(tip_name=lambda d: d["ID"].map(tips_by_id))
    [["entrytag_name", "Glottocode", "ID", "tip_name"]]
)

# ── 4. Join tip mapping back to entry level; take deepest matched level per entry ──

drh_asjp = drh_langs.merge(tip_map, on="entrytag_name", how="left")

entry_tip = (drh_asjp[drh_asjp["tip_name"].notna()]
             .sort_values("entrytag_level", ascending=False)
             .drop_duplicates("entry_id")
             [["entry_id", "entrytag_name", "Glottocode", "ID", "tip_name"]])

print(f"Entries mapped to a tree tip: {len(entry_tip)}")
print(f"Unique tips: {entry_tip['tip_name'].nunique()}")

# ── 5. Merge each mdl_input file with the tip mapping and save ──

os.makedirs("../data/phylo_input", exist_ok=True)

for f in mdl_files:
    name = os.path.basename(f)
    data = pd.read_csv(f)

    out = (data
           .merge(entry_tip[["entry_id", "tip_name", "ID", "Glottocode"]],
                  on="entry_id", how="inner")  # inner: drop entries without a tip
           .reset_index(drop=True))

    out.to_csv(f"../data/phylo_input/{name}", index=False)
    print(f"{name}: {len(data)} -> {len(out)} rows, {out['tip_name'].nunique()} unique tips")
