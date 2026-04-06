"""
vmp 2026-03-29
Map Religious Group entries to tips in the ASJP language phylogeny (Jäger 2018).

Pipeline:
  DRH entity tags -> Glottolog (name -> Glottocode) -> ASJP (Glottocode -> ID) -> world.tre (ID -> tip_name)

Input:  data/preprocessed/answerset.csv       (551 entries)
Output: data/preprocessed/tip_map.csv         (entry_id -> tip_name, one row per entry)
        data/preprocessed/tip_map_lost.csv    (entries lost at each stage, with reason)
"""

import pandas as pd
from Bio import Phylo

# ── 1. Entries we need to map ──────────────────────────────────────────────────

answerset = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = set(answerset["entry_id"].unique())
print(f"Entries to map: {len(entry_ids)}") # 458

# ── 2. Load DRH language tags ──────────────────────────────────────────────────
# before Matthew Hamm codings.

drh_tags  = pd.read_csv("../data/raw/entity_tags.csv")
drh_tags  = drh_tags[drh_tags["entry_id"].isin(entry_ids)]

drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]
drh_langs = drh_langs.drop_duplicates()
drh_langs = drh_langs.dropna(subset='entrytag_name')
drh_langs["Coding"] = "SCCSR.v3"
len(drh_langs) # 5730

no_tag = entry_ids - set(drh_langs["entry_id"])
print(f"No language tag (old):          {len(no_tag)}") 
print(f"With language tag (old):        {drh_langs['entry_id'].nunique()}")

# --- 2.1. recoded records ----
# Matthew Hamm codings.
coded_entries = pd.read_csv("../preprocessing/data/coded_entries.csv")
coded_entries = coded_entries[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]

# the ones that are not NA we already have
# or we need to recode in another way
coded_entries = coded_entries[coded_entries["entrytag_path"].isna()] # n=58
coded_entries["Coding"] = "Matthew Hamm"
drh_langs = pd.concat([drh_langs, coded_entries], ignore_index=True)
drh_langs = drh_langs.drop_duplicates() 
len(drh_langs) # 5788

no_tag = entry_ids - set(drh_langs["entry_id"])
print(f"No language tag (new):          {len(no_tag)}")
print(f"With language tag (new):        {drh_langs['entry_id'].nunique()}")

# ── 3. Load reference tables ───────────────────────────────────────────────────

glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["Glottocode", "Glottolog_Name"]

asjp = pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Glottocode"]].drop_duplicates()

tree     = Phylo.read("../data/jaeger2018/world.tre", "newick")
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
len(drh_mapped) # 135405

# ── 5. Document losses at each stage ──────────────────────────────────────────

# lost: tag present but no Glottolog/ASJP match
has_tag     = drh_mapped["entry_id"].unique()
matched_asjp = drh_mapped[drh_mapped["ID"].notna()]["entry_id"].unique()
lost_no_asjp = (
    drh_mapped[drh_mapped["entry_id"].isin(has_tag) & ~drh_mapped["entry_id"].isin(matched_asjp)]
    .sort_values("entrytag_level", ascending=False)
    .drop_duplicates("entry_id")
    [["entry_id", "entrytag_name", "entrytag_path", "Glottocode", "ID", "Coding"]]
    .assign(reason="no_asjp_match")
)

# lost: ASJP matched but ID not in tree
matched_tip = drh_mapped[drh_mapped["tip_name"].notna()]["entry_id"].unique()
lost_no_tip = (
    drh_mapped[drh_mapped["entry_id"].isin(matched_asjp) & ~drh_mapped["entry_id"].isin(matched_tip)]
    .sort_values("entrytag_level", ascending=False)
    .drop_duplicates("entry_id")
    [["entry_id", "entrytag_name", "entrytag_path", "Glottocode", "ID", "Coding"]]
    .assign(reason="not_in_tree")
)

entry_meta = (pd.read_csv("../data/raw/entry_data.csv")[["entry_id", "entry_name"]]
              .assign(drh_link=lambda d: "https://religiondatabase.org/browse/" + d["entry_id"].astype(str)))

lost = (pd.concat([lost_no_asjp, lost_no_tip], ignore_index=True)
        .merge(entry_meta, on="entry_id", how="left")
        .sort_values("entry_id")
        [["entry_id", "entry_name", "entrytag_name", "entrytag_path", "Glottocode", "Coding", "reason", "drh_link", "ID"]])


print(f"Lost — no ASJP match:     {len(lost_no_asjp)}")
print(f"Lost — not in tree:       {len(lost_no_tip)}")
print(f"Total lost:               {lost['entry_id'].nunique()}")

# ── 6. Final mapping: one tip per entry (deepest matched level) ────────────────

entry_tip = (
    drh_mapped[drh_mapped["tip_name"].notna()]
    .sort_values("entrytag_level", ascending=False)
    .drop_duplicates("entry_id")
    [["entry_id", "entrytag_name", "entrytag_path", "Glottocode", "ID", "tip_name", "Coding"]]
    .sort_values("entry_id")
    .reset_index(drop=True)
)

print(f"Mapped to tree tip:       {len(entry_tip)}")
print(f"Unique tips:              {entry_tip['tip_name'].nunique()}")

# ── 7. Save ────────────────────────────────────────────────────────────────────

entry_tip.to_csv("../data/preprocessed/tip_map.csv", index=False)
lost.to_csv("../data/preprocessed/tip_map_lost.csv", index=False)
