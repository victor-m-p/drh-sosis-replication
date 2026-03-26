import pandas as pd

# load
drh_tags = pd.read_csv("../data/raw/entity_tags.csv")

# only take the cultures that are relevant
drh_answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
drh_entries = drh_answers["entry_id"].unique()
drh_tags = drh_tags[drh_tags["entry_id"].isin(drh_entries)]
print(drh_tags["entry_id"].nunique()) # n=828

# take language rows
drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
print(drh_langs["entry_id"].nunique()) # n = 747
drh_langs = drh_langs[['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]
drh_langs_unique = drh_langs[['entrytag_name']].drop_duplicates()

# --- step 2: ASJP ---
asjp = pd.read_csv("../asjp/cldf/languages.csv")

asjp = asjp[
    ["ID", "Name", "Glottocode"]
].drop_duplicates()

# --- step 3: Glottolog ---
glottolog = pd.read_csv("../data/glottolog/languoid.csv")

glottolog = glottolog[
    ["id", "name"]
]

glottolog.columns = ["Glottocode", "Glottolog_Name"]

# --- step 4: merge Glottolog and DRH ---
drh_glottolog = pd.merge(
    drh_langs_unique,
    glottolog,
    left_on="entrytag_name",
    right_on="Glottolog_Name",
    how="left"
)

# --- step 5: now merge with ASJP ---
# this is where we are losing data. 
# but I am keeping many more now because of better merge.
drh_asjp = pd.merge(
    drh_glottolog,
    asjp,
    on="Glottocode",
    how="left"
)

# keep only rows that have an ASJP doculect
drh_asjp = drh_asjp[drh_asjp["ID"].notna()]

# merge back to the entry-level table
drh_asjp = drh_langs.merge(
    drh_asjp,
    on="entrytag_name",
    how="inner"
)

print(drh_asjp["entry_id"].nunique()) # 694 (much better.)

### --- copying in the previus pipeline ---
### now do the merge here and then filter out later to get the deepest level of the tree for each entry ###
from Bio import Phylo

## from: https://osf.io/cufv7/overview 
tree = Phylo.read("../asjp/raw/world.tre", "newick")
tips = [t.name for t in tree.get_terminals()]

# Tree tips are "FAMILY.SUBGROUP.ASJP_ID" — the ID is the last component
tips_by_id = {t.split('.')[-1]: t for t in tips}

# Map our DRH entries to their tree tip
drh_asjp["tip_name"] = drh_asjp["ID"].map(tips_by_id)
print(drh_asjp['entry_id'].nunique()) # 694 (still.)

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

# --- Selection: deepest level per entry that actually has a tree tip ---
# Filter to matched rows first, then take deepest — this naturally implements the fallback.
# Save the pre-selection frame so we can inspect what gets dropped.
drh_asjp_all = drh_asjp.copy() 
print(drh_asjp_all['entry_id'].nunique()) # 694 (still.)

drh_asjp = (drh_asjp[drh_asjp['tip_name'].notna()]
             .sort_values("entrytag_level", ascending=False)
             .drop_duplicates("entry_id")
             .sort_values('entry_id'))

# save matched entries
drh_asjp[['entry_id', 'entrytag_name', 'entrytag_level', 'ID', 'Glottocode', 'tip_name']].to_csv(
    "data/matched_v6.csv", index=False)

# --- Where are we losing entries? ---
lost_merge = (drh_langs[~drh_langs['entry_id'].isin(drh_asjp_all['entry_id'])]
              .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
              .drop_duplicates('entry_id')
              [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']])
len(lost_merge) # n=53

# 2) Lost at the ASJP ID -> tree tip step (565 -> 555)
#    These entries matched ASJP but none of their ASJP IDs appear in the tree.
#    Show the deepest ASJP match per entry (the one that would have been selected).
lost_tip = (drh_asjp_all[~drh_asjp_all['entry_id'].isin(drh_asjp['entry_id'])]
            .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
            .drop_duplicates('entry_id')
            [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path', 'ID', 'Glottocode']])
len(lost_tip) # 6

### how many of these tips are unique? ###
drh_asjp["tip_name"].nunique() # n=268
drh_asjp.groupby('tip_name').size().reset_index(name='count').sort_values('count', ascending=False).head(20)
drh_asjp.to_csv("data/drh_asjp.csv", index=False)

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

lost_records.to_csv("data/lost_records_v6.csv", index=False)

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
ritual_asjp['tip_name'].isna().sum() # 78 NA / 392

# which unmatched entries account for those 78 rows — grouped by entry_id
# so we can prioritise which to manually code (most rows lost = highest priority)
missing_tips = (ritual_asjp[ritual_asjp['tip_name'].isna()]
                .groupby('entry_id')
                .size()
                .reset_index(name='n_ritual_rows')
                .merge(entries[['entry_id', 'entry_name']], on='entry_id', how='left')
                .merge(drh_langs[['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]
                       .sort_values('entrytag_level', ascending=False)
                       .drop_duplicates('entry_id'),
                       on='entry_id', how='left')
                .assign(drh_link=lambda d: "https://religiondatabase.org/browse/" + d['entry_id'].astype(str))
                .sort_values('n_ritual_rows', ascending=False)
                [['entry_id', 'entry_name', 'n_ritual_rows',
                  'entrytag_name', 'entrytag_level', 'entrytag_path', 'drh_link']])

print(f"Unique unmatched entries: {len(missing_tips)}")
print(missing_tips.to_string(index=False))
missing_tips.to_csv("data/ritual_missing_tips.csv", index=False)

ritual_asjp = ritual_asjp.dropna()
ritual_asjp # 314 rows
ritual_asjp['tip_name'].nunique() # 144

# save for R phylogenetic model (entry-level, not aggregated)
ritual_asjp.to_csv("../data/phylo_input/ritual_asjp_phylo.csv", index=False)

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

# --- Comparison: v5 (old) vs v6 (new) ---
v5 = pd.read_csv("data/matched_v5.csv")
v6 = pd.read_csv("data/matched_v6.csv")

ids_v5 = set(v5['entry_id'])
ids_v6 = set(v6['entry_id'])

gained_ids = ids_v6 - ids_v5   # in v6 but not v5
dropped_ids = ids_v5 - ids_v6  # in v5 but not v6 (should be empty ideally)

print(f"v5 matched: {len(ids_v5)}  |  v6 matched: {len(ids_v6)}")
print(f"Gained in v6: {len(gained_ids)}  |  Dropped from v5: {len(dropped_ids)}")

# show gained entries with entry names
gained_df = (v6[v6['entry_id'].isin(gained_ids)]
             .merge(entries[['entry_id', 'entry_name']], on='entry_id', how='left')
             .sort_values('entry_id')
             [['entry_id', 'entry_name', 'entrytag_name', 'tip_name']])
print(f"\nEntries gained in v6 ({len(gained_df)}):")
print(gained_df.to_string(index=False))

# show any entries dropped (unexpected — flag for investigation)
if dropped_ids:
    dropped_df = (v5[v5['entry_id'].isin(dropped_ids)]
                  .merge(entries[['entry_id', 'entry_name']], on='entry_id', how='left')
                  .sort_values('entry_id')
                  [['entry_id', 'entry_name', 'entrytag_name', 'tip_name']])
    print(f"\nEntries dropped from v5 ({len(dropped_df)}) — investigate:")
    print(dropped_df.to_string(index=False))

# --- Symmetric difference: entries in exactly one of v5 / v6 ---
v5_only = (v5[v5['entry_id'].isin(dropped_ids)]
           .merge(entries[['entry_id', 'entry_name']], on='entry_id', how='left')
           .assign(present_in='v5_only'))

v6_only = (v6[v6['entry_id'].isin(gained_ids)]
           .merge(entries[['entry_id', 'entry_name']], on='entry_id', how='left')
           .assign(present_in='v6_only'))

sym_diff = (pd.concat([v5_only, v6_only])
            .sort_values(['present_in', 'entry_id'])
            [['entry_id', 'entry_name', 'entrytag_name', 'entrytag_level',
              'ID', 'Glottocode', 'tip_name', 'present_in']])

sym_diff.to_csv("data/v5_v6_difference.csv", index=False)
print(f"\nSymmetric difference saved: {len(sym_diff)} entries ({len(gained_ids)} v6-only, {len(dropped_ids)} v5-only)")