"""
vmp 2026-03-29 (scratch: provenance + precedence)
Map Religious Group entries to tips in the ASJP language phylogeny (Jäger 2018).
  DRH tags -> Glottolog (name->code) -> ASJP (code->ID) -> world.tre (ID->tip)
Precedence when an entry has candidates from several sources: manual_B > manual_A > DRH.
  - DRH      : automatic name-match through the chain
  - manual_A : expert re-name (Matthew Hamm), still resolved via the name-match
  - manual_B : expert coded the TIP directly (authoritative); its Glottocode/entrytag_name
               are the ORIGINAL values being corrected, so we do NOT trust them downstream.
"""

import pandas as pd
from pathlib import Path
from Bio import Phylo

OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
RANK = {"DRH": 1, "manual_A": 2, "manual_B": 3}   # higher = more authoritative

# 1. Entries we need to map
answerset = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = set(answerset["entry_id"].unique())
len(entry_ids) # 458 entries

# 2. DRH language tags
drh_tags  = pd.read_csv("../data/raw/entity_tags.csv")
drh_tags  = drh_tags[drh_tags["entry_id"].isin(entry_ids)]
drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs[drh_langs['entry_id']==667]

# also remove Level = 1 since that is just "Language"
drh_langs = drh_langs[drh_langs["entrytag_level"] > 1]
drh_langs = drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]
drh_langs = drh_langs.drop_duplicates().dropna(subset="entrytag_name")
drh_langs["coding"] = "SCCSR.v3"
drh_langs["source"] = "DRH" 
drh_langs['entry_id'].nunique() # 401 with language 

'''
intermezzo #1:
This means we have n=57 entries with no language tag at all.
These can only be resolved by manual coding.
'''

### intermezzo: which are these n=57 entries ###

# 2.1 manual A (expert re-name; resolves via name-match like DRH; carries coder in `coding`)
manual_codes_A = pd.read_csv("../data/raw/manual_lang_A.csv")
manual_codes_A["source"] = "manual_A" 
drh_langs = pd.concat([drh_langs, manual_codes_A], ignore_index=True).drop_duplicates()
drh_langs['entry_id'].nunique() # back to 458 entries (all).

'''
Codes the 57 entries with no DRH tag.
However, many of the names supplied here do not resolve to a tip.
'''

# 3. Reference tables
# Glottolog: name --> glottocode 
glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["Glottocode", "Glottolog_Name"]

# Jaeger 2018: ID --> tip_name
tree = Phylo.read("../data/jaeger2018/world.tre", "newick")
tree_tips = {t.name for t in tree.get_terminals()}  
tips_by_id = {t.name.split(".")[-1]: t.name for t in tree.get_terminals()} 
tip_last = set(tips_by_id) 
tree.count_terminals() # 6892

# ASJP: glottocode --> ID

'''
Before we had: 
asjp = (pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Glottocode"]].dropna(subset=["Glottocode"]).drop_duplicates("Glottocode"))
But this is not good because 1 glottocode can map to >1 ASJP.
We need to get the ASJP that is in the tree (if any).

So what we do now: 
We merge with the tree tips which narrows the ASJP set.
Still sometimes we now have multiple ASJP IDs for a single Glottocode on the tree.
In this case we need to choose one. 
Probably some can be basically automatically decided.
Some will probably need expert review.
We carry all the information along.
'''

# --- reference: glottocode -> glottolog language name (canonical) ---
gname = glottolog.set_index("Glottocode")["Glottolog_Name"]

def norm(s):
    return str(s).lower().replace("_", "").replace(" ", "")

asjp_full = (pd.read_csv("../asjp/cldf/languages.csv")[["ID","Glottocode"]]
             .dropna(subset=["Glottocode"]))
asjp_full["ID"] = asjp_full["ID"].astype(str)
asjp_full["in_tree"]  = asjp_full["ID"].isin(tip_last)
asjp_full["gname"]    = asjp_full["Glottocode"].map(gname)

# does the doculect ID match the Glottolog language name?
asjp_full["name_match"] = [norm(i) == norm(g) for i, g in
                           zip(asjp_full["ID"], asjp_full["gname"])]
asjp_full["idlen"] = asjp_full["ID"].str.len()

cands = (asjp_full[asjp_full["in_tree"]]
         .groupby("Glottocode")["ID"].apply(lambda s: sorted(s)))

# preference: (1) in tree, (2) ID matches the language name, (3) shortest
asjp = (asjp_full.sort_values(["in_tree","name_match","idlen"],
                              ascending=[False, False, True])
                 .drop_duplicates("Glottocode")[["ID","Glottocode"]])

def cand_str(row):
    c = cands.get(row["Glottocode"], [])
    if len(c) <= 1: return pd.NA
    others = [x for x in c if x != row["ID"]]
    return " | ".join([row["ID"]] + others)
asjp["asjp_candidates"] = asjp.apply(cand_str, axis=1)

# 4. Chain lookups on unique tag names (DRH + manual_A names)
tag_names = drh_langs[["entrytag_name"]].drop_duplicates()
len(tag_names) # 1180 to resolve.

# merge: DRH --> glottocode (m1)
m1 = tag_names.merge(glottolog, left_on="entrytag_name",
                     right_on="Glottolog_Name", how="left")
m1['Glottocode'].notna().sum() # 1130/1180 names resolved.

''' 
Many of the ones we lose here are recovered further up in the tree.
But some are not because they are the only one we assign e.g. from Matthew Hamm "Yoruba; English".

Actually this only rescues 17/57 that it codes into the ASJP.
So this step is unfortunately not as useful as I thought.
'''

# merge: glottocode --> ASJP ID (m2)
m2 = m1.merge(asjp, on="Glottocode", how="left")
m2["ID"].notna().sum() # 245/1180
len(m2) # still 1180 rows good. 
assert len(m2) == len(m1) # should stay same.

'''
N = 245 tags survive.
We are losing many tags here, but that is mostly at levels
that are not languages (e.g., further up the tree, or down).
We have 458 total entries and 297 resolve to ASJP here.
That means that 161 do not resolve.
'''

# merge 3: ID -> tip
m3 = m2.copy()
m3["tip_name"] = m3["ID"].map(tips_by_id)
len(m3) # still 1180 rows
m3["tip_name"].notna().sum() # 211
assert len(m3) == len(m2) # should stay same.

'''
N = 211 tags survive.
So here we lose another 34 tags that are not in the tree.
'''

# get the final tip map and map to DRH.
tip_map = (m3[["entrytag_name", "Glottocode", "ID", "tip_name", "asjp_candidates"]]
           .drop_duplicates("entrytag_name"))

drh_mapped = drh_langs.merge(tip_map, on="entrytag_name", how="left")

# entry-level resolution BEFORE manual_B (n=305)
drh_mapped.loc[drh_mapped["tip_name"].notna(), "entry_id"].nunique()

# Manual B: second round of manual coding # 
manual_codes_B = pd.read_csv("../data/raw/manual_lang_B.csv")
rows_B = (manual_codes_B[["entry_id", "entrytag_name", "tip_name", "coding", "fit"]]
          .assign(source="manual_B")
          .drop_duplicates())
drh_mapped = pd.concat([drh_mapped, rows_B], ignore_index=True)
drh_mapped.loc[drh_mapped["tip_name"].notna(), "entry_id"].nunique() # 366

# 5. One tip per entry: precedence B>A>DRH, then deepest level within a source
drh_mapped["source_rank"] = drh_mapped["source"].map(RANK)
assert drh_mapped["source_rank"].notna().all(), "row with unrecognised source"

entry_tip = (
    drh_mapped[drh_mapped["tip_name"].notna()]
    .sort_values(["source_rank", "entrytag_level"], ascending=[False, False])
    .drop_duplicates("entry_id")
    [["entry_id", "tip_name", "source", "coding", "fit",
      "entrytag_name", "entrytag_level", "entrytag_path",
      "Glottocode", "ID", "asjp_candidates", "source_rank"]]   # <-- asjp_candidates added
    .sort_values("entry_id")
    .reset_index(drop=True)
)

# 6. Split: mapped vs orphaned (no tip from ANY source)
mapped_ids = set(entry_tip["entry_id"])
orphan_ids = entry_ids - mapped_ids
deepest = (drh_langs.sort_values("entrytag_level", ascending=False)
                    .drop_duplicates("entry_id")
                    [["entry_id", "entrytag_name", "entrytag_level"]]
                    .rename(columns={"entrytag_name": "deepest_tag"}))
orphans = (pd.DataFrame({"entry_id": sorted(orphan_ids)})
             .merge(deepest, on="entry_id", how="left")
             .sort_values("entry_id"))

len(entry_tip) # n=366
len(orphans) # n=92

## for housekeeping, add entry name + select columns. ##
entry_data = pd.read_csv("../data/raw/entry_data.csv")
entry_data = entry_data[['entry_id', 'entry_name']].drop_duplicates()
entry_tip = entry_tip.merge(entry_data, on = 'entry_id', how = 'inner')
orphans = orphans.merge(entry_data, on = 'entry_id', how = 'inner')

## select columns 
entry_tip["NEEDS_CODE"] = entry_tip["asjp_candidates"].notna()
entry_tip["CODE"] = "" 
entry_tip["CODER"] = ""
entry_tip = entry_tip[["NEEDS_CODE", "CODE", "CODER", "entry_id", "entry_name", "entrytag_name", "tip_name", "Glottocode", "ID", "asjp_candidates", "source", "coding"]]
entry_tip = entry_tip.sort_values(["NEEDS_CODE", "entry_id"], ascending=[False, True])
orphans = orphans[["entry_id", "entry_name", "deepest_tag"]]

## a few prints
entry_tip.groupby(["NEEDS_CODE"]).size() 

'''
False: 281
True: 85
'''

entry_tip.groupby("source").size() 

'''
DRH: 288
manual_A: 17
manual_B: 61
'''

entry_tip.to_csv(OUT / "tip_map_mapped.csv", index=False)
orphans.to_csv(OUT / "tip_map_orphans.csv", index=False)
