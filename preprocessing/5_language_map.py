'''
Mapping to: 

https://www.nature.com/articles/sdata2018189

'''

import re
import pandas as pd

# load
tags = pd.read_csv("../data/raw/entity_tags.csv")

# only take the cultures that are relevant
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
answer_entries = answers["entry_id"].unique()
tags = tags[tags["entry_id"].isin(answer_entries)]
tags["entry_id"].nunique() # n=828

# take language rows
lang_rows = tags[tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
lang_rows["entry_id"].nunique() # n = 747
lang_rows = lang_rows[['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]

# now load the data from the paper
languages = pd.read_csv("../asjp/cldf/languages.csv")
languages = languages[["Glottocode", "Glottolog_Name"]].drop_duplicates() # n=6127 rows

# merge with the DRH data.
lang_merge = pd.merge(lang_rows, languages, left_on="entrytag_name", right_on="Glottolog_Name", how="left").dropna()
lang_merge['entry_id'].nunique() # n = 565 (so not all present.)

# take the deepest one for each entry_id 
# based on the entrytag_level column 
lang_merge = lang_merge.sort_values("entrytag_level", ascending=False).drop_duplicates("entry_id")
lang_merge['entry_id'].nunique() # n = 565 (same as before, so no duplicates)
lang_merge = lang_merge.sort_values('entry_id')
lang_merge

## next step: https://osf.io/cufv7/overview ##
# download the above and load the world.tree file
# see whether we have matching tips.
'''
from Bio import Phylo

tree = Phylo.read("world.tre", "newick")
tips = [t.name for t in tree.get_terminals()]
len(tips), tips[:10]
'''