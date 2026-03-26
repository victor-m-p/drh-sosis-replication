'''
Mapping to: 

https://www.nature.com/articles/sdata2018189

OSF for the article: https://osf.io/cufv7/files/osfstorage
Glottolog ASJP mapping: https://asjp.clld.org/languages 

'''

import pandas as pd

# load
drh_tags = pd.read_csv("../data/raw/entity_tags.csv")

# only take the cultures that are relevant
drh_answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
drh_entries = drh_answers["entry_id"].unique()
drh_tags = drh_tags[drh_tags["entry_id"].isin(drh_entries)]
drh_tags["entry_id"].nunique() # n=828

# take language rows
drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs["entry_id"].nunique() # n = 747
drh_langs = drh_langs[['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]

# now load the data from the paper
asjp_langs = pd.read_csv("../asjp/cldf/languages.csv")
asjp_langs = asjp_langs[["ID", "Glottocode", "Glottolog_Name"]].drop_duplicates()
asjp_langs # 11540 (6127 unique Glottocode.)

# merge the two datasets.
drh_asjp = pd.merge(drh_langs, asjp_langs, left_on="entrytag_name", right_on="Glottolog_Name", how="left").dropna()
drh_asjp['entry_id'].nunique() # n = 565 (so not all present.)

### now do the merge here and then filter out later to get the deepest level of the tree for each entry ###
from Bio import Phylo

## from: https://osf.io/cufv7/overview 
tree = Phylo.read("../asjp/raw/world.tre", "newick")
tips = [t.name for t in tree.get_terminals()]

# Tree tips are "FAMILY.SUBGROUP.ASJP_ID" — the ID is the last component
tips_by_id = {t.split('.')[-1]: t for t in tips}

# Check overall ASJP <-> tree coverage
print(f"ASJP IDs in tree:  {len(set(asjp_langs['ID']) & set(tips_by_id))}")  # ~6,812 of 11,540

# Map our DRH entries to their tree tip
drh_asjp['tip_name'] = drh_asjp['ID'].map(tips_by_id)
print(f"DRH entries with a tree tip: {drh_asjp['tip_name'].notna().sum()} / {len(drh_asjp)}")

# --- Diagnostic: does falling back to shallower levels ever rescue an entry? ---
# The naive approach (sort by level, drop_duplicates) selects the deepest ASJP-matched
# tag per entry, regardless of whether it has a tree tip.
# here we instead find the deepest ASJP match per entry, 
# - then check how many of those have a tree tip
# — and for those that don't
# - check if any shallower match does have a tip.
# - take the deepest possible with a tip.
naive = drh_asjp.sort_values("entrytag_level", ascending=False).drop_duplicates("entry_id")
naive_no_tip = naive[naive['tip_name'].isna()]['entry_id']

# For those "failed" entries, check if any shallower ASJP match does have a tip
rescued = drh_asjp[
    drh_asjp['entry_id'].isin(naive_no_tip) &
    drh_asjp['tip_name'].notna()
]
#print(f"Entries where deepest ASJP match lacks a tree tip: {naive_no_tip.nunique()}")
#print(f"Of those, entries rescued by a shallower level:    {rescued['entry_id'].nunique()}")
if not rescued.empty:
    print(rescued[['entry_id', 'entrytag_name', 'entrytag_level', 'tip_name']].to_string())

# --- Selection: deepest level per entry that actually has a tree tip ---
# Filter to matched rows first, then take deepest — this naturally implements the fallback.
# Save the pre-selection frame so we can inspect what gets dropped.
drh_asjp_all = drh_asjp.copy()  # 565 entries, tip_name may be NaN

drh_asjp = (drh_asjp[drh_asjp['tip_name'].notna()]
             .sort_values("entrytag_level", ascending=False)
             .drop_duplicates("entry_id")
             .sort_values('entry_id'))

# save matched entries for comparison with v6
drh_asjp[['entry_id', 'entrytag_name', 'entrytag_level', 'ID', 'Glottocode', 'tip_name']].to_csv(
    "data/matched_v5.csv", index=False)

# --- Where are we losing entries? ---

# 1) Lost at the Glottolog_Name -> ASJP merge (747 -> 565)
#    These entries had language tags but none of their tag names matched any
#    Glottolog_Name in languages.csv. Show the deepest tag per entry (best attempt).
#pd.set_option('display.max_colwidth', None)
lost_merge = (drh_langs[~drh_langs['entry_id'].isin(drh_asjp_all['entry_id'])]
              .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
              .drop_duplicates('entry_id')
              [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']])

# 2) Lost at the ASJP ID -> tree tip step (565 -> 555)
#    These entries matched ASJP but none of their ASJP IDs appear in the tree.
#    Show the deepest ASJP match per entry (the one that would have been selected).
lost_tip = (drh_asjp_all[~drh_asjp_all['entry_id'].isin(drh_asjp['entry_id'])]
            .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
            .drop_duplicates('entry_id')
            [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path', 'ID', 'Glottocode']])

### how many of these tips are unique? ###
drh_asjp["tip_name"].nunique() # only 203/555 
drh_asjp.groupby('tip_name').size().reset_index(name='count').sort_values('count', ascending=False).head(20)

'''

ASJP: 
- based on https://en.wikipedia.org/wiki/Swadesh_list (only 40 words.)
- did not put glottocode for Egyptian e.g. (middle).
- potential task: can we save some of these?

Seems like yes e.g. we have Greek and they have ancient Greek.
Some of our entries do not go deep enough e.g.:
- Greek vs. Ancient Greek.


https://wals.info/
https://asjp.clld.org/languages
https://glottolog.org/resource/languoid/id/anci1242

Can search on ASJP and put in code on glottolog.

Provide: 
- ask for a tip glottocode (what is tip?)
- provide entry name + link. 

# 1. how do we find tips 
# 2. do we need tips
'''

#### save document for Willis ####
# first merge lost_tip and lost_merge and add a column
lost_tip['reason'] = 'tip'
lost_merge['reason'] = 'merge'
lost_records = pd.concat([lost_merge, lost_tip])

# then we need to get the entry name
entries = pd.read_csv("../data/preprocessed/entries_clean.csv")
entries = entries[['entry_id', 'entry_name', 'data_source']].drop_duplicates()
lost_records = lost_records.merge(entries, on = 'entry_id', how = 'inner')

# now add the link 
lost_records['drh_link'] = (
    "https://religiondatabase.org/browse/" 
    + lost_records['entry_id'].astype(str)
)

# select columns in reasonable order:
lost_records = lost_records[[
    "entry_id",
    "entry_name",
    "entrytag_name",
    "entrytag_level",
    "entrytag_path",
    "drh_link",
    "data_source",
    "reason",
    "ID",
    "Glottocode"]]

lost_records.to_csv("data/lost_records_v5.csv", index=False)

''' Questions:
1. Why are so many entries lost on merge? 
--> importantly: any pattern in this? 
--> this could be problematic if yes. 
--> can we find some of them? or at least figure out why?
2. What do we do about duplicate tips? 
3. Can we use this tree for our analysis? (asjp)
4. Where are the Glottocodes in DRH? (how do I get them?)

# How I think this works: 
1. BayesTraits takes a data file where each row is a taxon name matching exactly 1 tip label in the tree. 
2. This means we must aggregate DRH entries to one value per tip (203 data points currently).
- we need to figure out how to do the aggregation e.g.: 
--> if all entries at a tip agree use that value
--> if they conflict take majority or code as missing.
--> but the above seems not optimal actually (in a Bayesian sense).
3. Then we must prune the world tree to only the 203 tips we have data for. 

# Notes
1. Single tree: e.g., world.tre is simpler and treats phylogeny as known truth.
2. Posterior sample of trees: accounts for phylogenetic uncertainty and could be preferred.

So in the end 203 is our effective sample size.
This might be okay since we already have the other analyses.
'''

# check how aggregation would look (example, extra ritual)
data_ritual = pd.read_csv("../data/mdl_input/extra_ritual_group_markers.csv")
drh_asjp_sub = drh_asjp[['entry_id', 'ID', 'tip_name']]
ritual_asjp = pd.merge(data_ritual, drh_asjp_sub, on="entry_id", how="left")
ritual_asjp['tip_name'].isna().sum() # 152 NA / 392
ritual_asjp = ritual_asjp.dropna()
ritual_asjp # 240 rows 
ritual_asjp['tip_name'].nunique() # 101 unique tips

'''
There is apparently a way to not do the aggreagation,
but tried to develop it below. More modern might be to do it in R.
'''

# aggregation on the 101 unique tips:
# Strategy: majority vote per tip.
# - unanimous (all 0 or all 1): use that value
# - majority (>50%): use majority
# - tied (exactly 50%): code as missing — BayesTraits accepts "?" for missing

def majority_vote(s):
    m = s.mean()
    if m > 0.5: return 1
    if m < 0.5: return 0
    return float('nan')  # tie → missing

def conflict_label(s):
    m = s.mean()
    if m in (0.0, 1.0): return 'unanimous'
    if m == 0.5:        return 'tied'
    return 'majority'

# diagnostic: how often do entries at the same tip disagree?
for col in ['violent_external', 'extra_ritual_group_markers']:
    conflict = ritual_asjp.groupby('tip_name')[col].apply(conflict_label)
    print(f"\n{col}:")
    print(conflict.value_counts().to_string())

# aggregate to one row per tip
agg = ritual_asjp.groupby('tip_name').agg(
    violent_external        = ('violent_external',             majority_vote),
    extra_ritual_group_markers = ('extra_ritual_group_markers', majority_vote),
    n_entries               = ('entry_id', 'count'),
).reset_index().sort_values('n_entries', ascending=False)

agg.head(20)

print(f"\nTips after aggregation: {len(agg)}")
print(f"Missing violent_external:           {agg['violent_external'].isna().sum()}")
print(f"Missing extra_ritual_group_markers: {agg['extra_ritual_group_markers'].isna().sum()}")

'''
# BayesTraits input: tab-separated, "?" for missing, tip_name first
bt = agg[['tip_name', 'violent_external', 'extra_ritual_group_markers']].copy()
bt['violent_external']             = bt['violent_external'].apply(lambda x: '?' if pd.isna(x) else str(int(x)))
bt['extra_ritual_group_markers']   = bt['extra_ritual_group_markers'].apply(lambda x: '?' if pd.isna(x) else str(int(x)))
bt.to_csv("../data/mdl_input/bayestraits_extra_ritual.tsv", sep='\t', index=False, header=False)
print("\nSample BayesTraits input:")
print(bt.head(10).to_string(index=False))
'''