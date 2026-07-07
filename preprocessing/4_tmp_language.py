import pandas as pd
from Bio import Phylo

mapped  = pd.read_csv("data/tip_map_mapped.csv")
orphans = pd.read_csv("data/orphans_to_confirm.csv")
tree = Phylo.read("../data/jaeger2018/world.tre", "newick")
tree_tips = {t.name for t in tree.get_terminals()}

# n=366 (from mapped)
map_part = mapped[["entry_id","entry_name","tip_name"]].copy()

# n=72
orphans = orphans[orphans['PROPOSED_tip'].notna()]
orphans = orphans.rename(columns={"PROPOSED_tip": "tip_name"})
orphans = orphans[["entry_id", "entry_name", "tip_name"]]

# combine (n=438)
combined = pd.concat([map_part, orphans], ignore_index=True)

# sanity checks (pass)
assert combined['entry_id'].is_unique 
phantom = set(combined["tip_name"]) - tree_tips
assert not phantom, f"tip(s) not in tree: {phantom}"
combined.to_csv("data/tip_map.csv", index=False)


### ... ###
combined = pd.read_csv("data/tip_map.csv")
tips_used = sorted(combined["tip_name"].unique())
print("entries:", len(combined), "| distinct tips used:", len(tips_used))

counts = combined["tip_name"].value_counts()
print("tips:", len(counts), "| entries:", counts.sum())
print("mean per tip:", round(counts.mean(),2), "| median:", int(counts.median()), "| max:", counts.max())
print("\nsingletons (1 entry) :", (counts==1).sum())
print("tips with >=5 entries:", (counts>=5).sum())
print("\ntop 15 most-loaded tips:")
print(counts.head(15).to_string())

combined = pd.read_csv("data/tip_map.csv")
eng = combined[combined.tip_name=="IE.GERMANIC.ENGLISH"]
# join to answerset for region/date/outcome to see if they're clustered
ans = pd.read_csv("../data/preprocessed/answerset.csv")
e = eng.merge(ans, on="entry_id", how="left")
print("English entries:", len(e))
print("\nby region:"); print(e["world_region"].value_counts().head().to_string())
print("\nyear range:", e["year_from"].min(), e["year_from"].max())