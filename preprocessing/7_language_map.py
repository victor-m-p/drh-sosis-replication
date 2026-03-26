import glob
import pandas as pd
from Bio import Phylo

# --- step 1: collect all entry_ids that appear in any mdl_input file ---
# These are the entries we actually care about for the analysis.
mdl_files = glob.glob("../data/mdl_input/*.csv")
mdl_entries = set()
for f in mdl_files:
    df = pd.read_csv(f)
    if 'entry_id' in df.columns:
        mdl_entries.update(df['entry_id'].unique())

print(f"Unique entries across all mdl_input files: {len(mdl_entries)}")

# --- step 2: load DRH tags, filter to relevant entries only ---
drh_tags = pd.read_csv("../data/raw/entity_tags.csv")
drh_tags = drh_tags[drh_tags["entry_id"].isin(mdl_entries)]
print(f"Entries with any tag:          {drh_tags['entry_id'].nunique()}")

drh_langs = drh_tags[drh_tags["entrytag_path"].astype(str).str.startswith("Language[")].copy()
drh_langs = drh_langs[['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]
print(f"Entries with a language tag:   {drh_langs['entry_id'].nunique()}")

drh_langs_unique = drh_langs[['entrytag_name']].drop_duplicates()

# --- step 3: ASJP ---
asjp = pd.read_csv("../asjp/cldf/languages.csv")[["ID", "Name", "Glottocode"]].drop_duplicates()

# --- step 4: Glottolog ---
glottolog = pd.read_csv("../data/glottolog/languoid.csv")[["id", "name"]]
glottolog.columns = ["Glottocode", "Glottolog_Name"]

# --- step 5: DRH names → Glottocode (via Glottolog) ---
drh_glottolog = pd.merge(
    drh_langs_unique,
    glottolog,
    left_on="entrytag_name",
    right_on="Glottolog_Name",
    how="left"
)

# --- step 6: Glottocode → ASJP ID ---
drh_asjp = pd.merge(drh_glottolog, asjp, on="Glottocode", how="left")
drh_asjp = drh_asjp[drh_asjp["ID"].notna()]

# merge back to entry level
drh_asjp = drh_langs.merge(drh_asjp, on="entrytag_name", how="inner")
print(f"Entries matched to ASJP:       {drh_asjp['entry_id'].nunique()}")

# --- step 7: map ASJP ID → world.tre tip ---
tree = Phylo.read("../asjp/raw/world.tre", "newick")
tips = [t.name for t in tree.get_terminals()]
tips_by_id = {t.split('.')[-1]: t for t in tips}

drh_asjp["tip_name"] = drh_asjp["ID"].map(tips_by_id)

# fallback logic: take deepest level that has a tip
drh_asjp_all = drh_asjp.copy()

drh_asjp = (drh_asjp[drh_asjp['tip_name'].notna()]
             .sort_values("entrytag_level", ascending=False)
             .drop_duplicates("entry_id")
             .sort_values('entry_id'))

print(f"Entries matched to tree tip:   {drh_asjp['entry_id'].nunique()}")
print(f"Unique tree tips:              {drh_asjp['tip_name'].nunique()}")

# --- step 8: lost entries ---
entries = pd.read_csv("../data/preprocessed/entries_clean.csv")
entries = entries[['entry_id', 'entry_name', 'data_source']].drop_duplicates()

# 1) no language tag at all
lost_language = (pd.DataFrame({'entry_id': list(mdl_entries - set(drh_langs['entry_id']))})
                 .assign(reason='language'))

# 2) language tag present but no ASJP match
lost_merge = (drh_langs[~drh_langs['entry_id'].isin(drh_asjp_all['entry_id'])]
              .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
              .drop_duplicates('entry_id')
              [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path']]
              .assign(reason='merge'))

# 3) ASJP matched but ID not in tree
lost_tip = (drh_asjp_all[~drh_asjp_all['entry_id'].isin(drh_asjp['entry_id'])]
            .sort_values(['entry_id', 'entrytag_level'], ascending=[True, False])
            .drop_duplicates('entry_id')
            [['entry_id', 'entrytag_name', 'entrytag_level', 'entrytag_path', 'ID', 'Glottocode']]
            .assign(reason='tip'))

lost_records = (pd.concat([lost_language, lost_merge, lost_tip])
                .merge(entries, on='entry_id', how='left')
                .assign(drh_link=lambda d: "https://religiondatabase.org/browse/" + d['entry_id'].astype(str)))

lost_records = lost_records[[
    'entry_id', 'entry_name', 'entrytag_name', 'entrytag_level',
    'entrytag_path', 'drh_link', 'data_source', 'reason', 'ID', 'Glottocode']]

print(f"\nLost — no language tag: {lost_language['entry_id'].nunique()}")
print(f"Lost — no ASJP match:   {lost_merge['entry_id'].nunique()}")
print(f"Lost — not in tree:     {lost_tip['entry_id'].nunique()}")
print(f"Total lost:             {lost_records['entry_id'].nunique()}")

# --- step 9: save ---
drh_asjp[['entry_id', 'entrytag_name', 'entrytag_level', 'ID', 'Glottocode', 'tip_name']].to_csv(
    "data/matched_v7.csv", index=False)
lost_records.to_csv("data/lost_records_v7.csv", index=False)
drh_asjp.to_csv("data/drh_asjp_v7.csv", index=False)

### check a few (language)
# yes, checks out.
drh_tags[drh_tags["entry_id"]==1667]