"""
Explore ASJP candidates for lost entries (no_asjp_match).
All have a Glottocode at family/macro level with no direct ASJP entry.
Strategy: walk the Glottolog parent chain upward from each in-tree ASJP language
to find which ones descend from the lost Glottocodes.
Output: preprocessing/data/lost_candidates.csv
         preprocessing/data/asjp_in_tree.csv
"""

import pandas as pd
from Bio import Phylo

# ── 1. Load inputs ─────────────────────────────────────────────────────────────

lost = pd.read_csv("../data/preprocessed/tip_map_lost.csv")
lost = lost[lost["reason"] == "no_asjp_match"][["entry_id", "entry_name", "entrytag_name", "Glottocode"]].copy()

glottolog = pd.read_csv("../data/glottolog/languoid.csv")[
    ["id", "name", "level", "parent_id"]
].set_index("id")

asjp_full = pd.read_csv("../asjp/cldf/languages.csv")[
    ["ID", "Name", "Glottocode", "Glottolog_Name", "ISO639P3code", "Macroarea",
     "Latitude", "Longitude", "Family", "classification_glottolog",
     "recently_extinct", "long_extinct"]
].drop_duplicates()

asjp = asjp_full[["ID", "Name", "Glottocode"]].dropna(subset=["Glottocode"])

tree = Phylo.read("../data/jaeger2018/world.tre", "newick")
tips_by_id = {t.name.split(".")[-1]: t.name for t in tree.get_terminals()}

# restrict ASJP to languages that are in the tree
asjp_in_tree = asjp[asjp["ID"].isin(tips_by_id)]

# ── 1b. Save full reference: all ASJP languages present in the tree ────────────

asjp_tree_ref = (asjp_full[asjp_full["ID"].isin(tips_by_id)]
                 .assign(tip_name=lambda d: d["ID"].map(tips_by_id))
                 .sort_values("ID")
                 .reset_index(drop=True))
asjp_tree_ref.to_csv("../preprocessing/data/asjp_in_tree.csv", index=False)

# ── 2. Build ancestor sets by walking parent_id chain up from each ASJP language

def get_ancestors(gc, max_depth=20):
    ancestors = set()
    current = gc
    for _ in range(max_depth):
        if current not in glottolog.index:
            break
        parent = glottolog.at[current, "parent_id"]
        if pd.isna(parent) or parent == current:
            break
        ancestors.add(parent)
        current = parent
    return ancestors

asjp_ancestors = {
    row["ID"]: get_ancestors(row["Glottocode"])
    for _, row in asjp_in_tree.iterrows()
}

# ── 3. For each lost Glottocode, find descendant ASJP languages in the tree ───

target_gcs = lost["Glottocode"].dropna().unique()

rows = []
for gc in sorted(target_gcs):
    entries = lost[lost["Glottocode"] == gc]
    descendants = asjp_in_tree[
        asjp_in_tree["ID"].apply(lambda i: gc in asjp_ancestors.get(i, set()))
    ]
    for _, row in entries.iterrows():
        rows.append({
            "entry_id":      row["entry_id"],
            "entry_name":    row["entry_name"],
            "entrytag_name": row["entrytag_name"],
            "Glottocode":    gc,
            "n_candidates": len(descendants),
            "candidates":   "; ".join(
                f"{r['ID']} ({r['Name']})" for _, r in descendants.iterrows()
            )
        })

# ── 4. Save and print summary ──────────────────────────────────────────────────

df = pd.DataFrame(rows).sort_values(["Glottocode", "entry_id"])
df.to_csv("../preprocessing/data/lost_candidates.csv", index=False)

summary = (df.groupby("Glottocode")
             .agg(n_entries=("entry_id", "count"), n_candidates=("n_candidates", "first"))
             .reset_index()
             .sort_values("n_candidates", ascending=False))

print(summary.to_string(index=False))