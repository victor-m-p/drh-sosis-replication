import pandas as pd
from Bio import Phylo

# tree tips -> asjp_id (last dot-token of the tip name)
tips = [t.name for t in Phylo.read("../data/jaeger2018/world.tre", "newick").get_terminals()]
ref = pd.DataFrame({"tip_name": tips})
ref["asjp_id"] = ref["tip_name"].str.split(".").str[-1]

# ASJP: ID -> Glottocode (keep all; multiple ASJP per code stay as separate rows)
asjp = pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Glottocode"]].drop_duplicates("ID")
asjp["ID"] = asjp["ID"].astype(str)
ref = ref.merge(asjp, left_on="asjp_id", right_on="ID", how="left").drop(columns="ID")

# Glottolog: code -> name
glot = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glot.columns = ["Glottocode", "glottolog_name"]
ref = ref.merge(glot, on="Glottocode", how="left")

ref = ref[["glottolog_name", "Glottocode", "asjp_id", "tip_name"]]
ref.to_csv("data/tree_reference.csv", index=False)
