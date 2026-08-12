"""
Prepare manual coding round C.
Already has manual coding rounds A, B.
This can be skipped given the now existing codings.
Keeping this for reference.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from Bio import Phylo

RAW = Path("../data/raw")
OUT = Path("data"); OUT.mkdir(parents=True, exist_ok=True)
RANK = {"DRH": 1, "manual_A": 2, "manual_B": 3} # higher = more authoritative

# entries + tags
answerset = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = set(answerset["entry_id"].unique())

drh_tags  = pd.read_csv(RAW / "entity_tags.csv")
drh_tags  = drh_tags[drh_tags["entry_id"].isin(entry_ids)]
drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[drh_langs["entrytag_level"] > 1] # drop the bare "Language" root
drh_langs = (drh_langs[["entry_id", "entrytag_name", "entrytag_level", "entrytag_path"]]
             .drop_duplicates().dropna(subset="entrytag_name"))
drh_langs["source"] = "DRH"
drh_langs["coder_v1"] = pd.NA # DRH tags have no single coder

# manual_A: expert assign new entry names in round A of coding.
# resolves through the same name --> code chain as DRH. 
manual_A = pd.read_csv(RAW / "manual_lang_A.csv").rename(columns={"coding": "coder_v1"})
manual_A["source"] = "manual_A"

name_tags = pd.concat([drh_langs, manual_A], ignore_index=True).drop_duplicates()

# The ORIGINAL DRH tag per entry (deepest recorded DRH language tag). Empty where none exists.
drh_original = (drh_langs.sort_values("entrytag_level", ascending=False)
                .drop_duplicates("entry_id")[["entry_id", "entrytag_name"]]
                .rename(columns={"entrytag_name": "orig_tag"}))

# reference tables
# glottolog: name -> code (authoritative name resolution)
glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["glottocode", "glottolog_name"]
gname = glottolog.drop_duplicates("glottocode").set_index("glottocode")["glottolog_name"]

# ASJP languages.csv: code -> doculect ID (+ its own Glottolog name for verification)
asjp = pd.read_csv("../data/asjp/languages.csv")[["ID", "Glottocode", "Glottolog_Name"]]
asjp = asjp.rename(columns={"ID": "asjp_id", "Glottocode": "glottocode",
                            "Glottolog_Name": "asjp_glottolog_name"})
asjp["asjp_id"] = asjp["asjp_id"].astype(str)
assert asjp["asjp_id"].is_unique

# Jaeger 2018 tree
tree = Phylo.read("../data/jaeger2018/world.tre", "newick")
tip_names = [t.name for t in tree.get_terminals()]

# tree maps + check whether any of the last parts of the tip names are not unique.
# ASJP doculect ID == final dotted field of a tip label, e.g. NC.BANTU.ZULU_2 -> ZULU_2
tips_by_last = defaultdict(list)
for name in tip_names:
    tips_by_last[name.split(".")[-1]].append(name)

collisions = {k: v for k, v in tips_by_last.items() if len(v) > 1}
assert len(collisions) == 0
tip_of_id = {k: v[0] for k, v in tips_by_last.items()} # safe when no collision

asjp["in_tree"] = asjp["asjp_id"].isin(tips_by_last)
asjp["tip_name"] = asjp["asjp_id"].map(tip_of_id)

# CHECK B (reverse) + reusable table
id_to_code = asjp.set_index("asjp_id")["glottocode"]
tip_rev = pd.DataFrame({"tip_name": tip_names})
tip_rev["asjp_id"] = tip_rev["tip_name"].str.split(".").str[-1]
tip_rev["glottocode"] = tip_rev["asjp_id"].map(id_to_code)
tip_rev["glottolog_name"] = tip_rev["glottocode"].map(gname)
tip_rev.to_csv(OUT / "tip_to_glottocode.csv", index=False)

# name -> glottocode
'''
This step is always unique (one name --> one glottocode).
A few names (e.g., "Unknown") do not map, otherwise easy.
'''

code_by_name = glottolog.groupby("glottolog_name")["glottocode"].apply(lambda s: sorted(set(s)))
names = name_tags[["entrytag_name"]].drop_duplicates().copy()
names["glottocode_cands"] = names["entrytag_name"].map(code_by_name)
names["glottocode"] = names["glottocode_cands"].apply(
    lambda c: c[0] if isinstance(c, list) and len(c) == 1 else pd.NA)
len(code_by_name[code_by_name.str.len() > 1]) 

# code -> asjp -> tips
'''
The more problematic merge where two failure modes happen:
1. one glottocode --> multiple ASJP/tips
2. one glottocode --> no match in ASJP/tips
'''

asjp_tree = asjp[asjp["in_tree"]]
cand_by_code = (asjp_tree.groupby("glottocode")
                .agg(asjp_ids =("asjp_id",  lambda s: sorted(s)),
                     tip_names=("tip_name", lambda s: sorted(s)))
                .reset_index())
names = names.merge(cand_by_code, on="glottocode", how="left")
names["glottolog_name"] = names["glottocode"].map(gname)

# attach to entries
resolved = name_tags.merge(names, on="entrytag_name", how="left")
resolved = resolved.rename(columns={"entrytag_name": "entrytag_used"})
resolved["confidence_v1"] = pd.NA # DRH / manual_A carry no fit rating

# manual_B (authoritative tips)
# In the file, entrytag_name and Glottocode are the ORIGINAL (wrong) values being corrected.
# The expert assigned tip_name directly; back-infer glottocode/name FROM the tip.
manual_B = pd.read_csv(RAW / "manual_lang_B.csv")   # entry_id, entrytag_name, Glottocode, tip_name, coding, fit
mb = manual_B.rename(columns={"coding": "coder_v1", "fit": "confidence_v1"}).copy()
mb["source"] = "manual_B"
mb["entrytag_level"] = 99 # any high number to make it deepest level
mb["asjp_id"] = mb["tip_name"].astype(str).str.split(".").str[-1]
mb["glottocode"] = mb["asjp_id"].map(id_to_code) # back-inferred, authoritative
mb["glottolog_name"] = mb["glottocode"].map(gname)
mb["asjp_ids"] = mb["asjp_id"].apply(lambda a: [a] if pd.notna(a) else [])
mb["tip_names"] = mb["tip_name"].apply(lambda t: [t] if pd.notna(t) else [])
mb["glottocode_cands"] = pd.NA
mb["entrytag_used"] = pd.NA # no NAME was used; tip assigned directly

# ORIGINAL DRH tag recorded on the manual_B row (fallback for entries lacking a DRH tag)
mb_original = manual_B[["entry_id", "entrytag_name"]].rename(columns={"entrytag_name": "orig_tag_mb"})

# ---------------------------------------------------------------- combine + one row per entry
COLS = ["entry_id", "source", "coder_v1", "confidence_v1",
        "entrytag_used", "entrytag_level",
        "glottolog_name", "glottocode", "glottocode_cands", "asjp_ids", "tip_names"]
resolved = resolved.reindex(columns=COLS)
mb = mb.reindex(columns=COLS)

combined = pd.concat([resolved[COLS], mb[COLS]], ignore_index=True)
for c in ("asjp_ids", "tip_names"):
    combined[c] = combined[c].apply(lambda x: x if isinstance(x, list) else [])

combined["n_tips"] = combined["tip_names"].apply(len)
combined["source_rank"] = combined["source"].map(RANK)
combined["resolved"] = combined["n_tips"] > 0

# per entry: prefer a resolved row, then higher precedence, then deepest tag
pick = (combined
        .sort_values(["resolved", "source_rank", "entrytag_level"],
                     ascending=[False, False, False])
        .drop_duplicates("entry_id")
        .copy())

# reindex over ALL entries so we can show entries that produced nothing
pick = pick.set_index("entry_id").reindex(sorted(entry_ids)).reset_index()

def status(r):
    n = len(r["tip_names"]) if isinstance(r["tip_names"], list) else 0
    if n == 1: return "resolved_unique"
    if n > 1:  return "resolved_multiple" # expert picks one
    if pd.notna(r["glottocode"]): return "glottocode_no_tree_tip"
    c = r["glottocode_cands"]
    if isinstance(c, list) and len(c) > 1: return "ambiguous_name"
    if pd.isna(r["entrytag_used"]): return "no_language_tag"
    return "name_not_in_glottolog"

pick["resolution_status"] = pick.apply(status, axis=1)
pick["n_candidates"] = pick["tip_names"].apply(lambda x: len(x) if isinstance(x, list) else 0)

# format + write expert sheet
entry_data = pd.read_csv(RAW / "entry_data.csv")[["entry_id", "entry_name"]].drop_duplicates()
out = pick.merge(entry_data, on="entry_id", how="left")

# ORIGINAL DRH tag: DRH deepest, falling back to the manual_B-recorded original
out = out.merge(drh_original, on="entry_id", how="left")
out = out.merge(mb_original,  on="entry_id", how="left")
out["entrytag_name"] = out["orig_tag"].fillna(out["orig_tag_mb"])
out = out.drop(columns=["orig_tag", "orig_tag_mb"])

# formatting
def join_list(x):
    return ", ".join(map(str, x)) if isinstance(x, list) and x else pd.NA
for c in ("asjp_ids", "tip_names", "glottocode_cands"):
    out[c] = out[c].apply(join_list)

# blank columns for the NEW authoritative coding round
out["tip_coder"] = ""
out["tip_assigned"] = ""
out["tip_confidence"] = ""
out["tip_comment"] = ""

out = out.rename(columns={"glottocode": "glottolog_id",
                          "asjp_ids": "asjp_id",
                          "tip_names": "jaeger_id"})
out = out[["entry_id", "entry_name",
           "entrytag_name", "entrytag_used",
           "glottolog_name", "glottolog_id", "asjp_id", "jaeger_id", "glottocode_cands",
           "resolution_status", "n_candidates",
           "source", "coder_v1", "confidence_v1",
           "tip_coder", "tip_assigned", "tip_confidence", "tip_comment"]]

print("\n[status counts]\n", out["resolution_status"].value_counts())
print("\n[source counts]\n", out["source"].value_counts(dropna=False))
out.to_csv(OUT / "tip_map_for_experts.csv", index=False)
