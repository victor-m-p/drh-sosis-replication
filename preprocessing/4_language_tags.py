'''
Look into language tags.
A couple of key things: 
- Entries have multiple language tags (rows).
- These are not always "contained" within each other.
- So both a question of (1) how deep, and (2) which one / all.
- This I do not know quite enough about.
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
lang_rows["entry_id"].nunique() # n=747

### figure out which ones are missing language ###

node_re = re.compile(r".+\[\d+\]$")  # sanity: each node looks like Name[123]

def path_depth(p):
    parts = str(p).split("->")
    # sanity check format (optional but useful)
    for x in parts:
        assert node_re.match(x.strip()), f"bad node format in path: {p}"
    return len(parts)

lang_rows["depth"] = lang_rows["entrytag_path"].map(path_depth)

# look at how specific tags are
print("LANGUAGE rows (including duplicates across entry_id):", len(lang_rows))
print("Unique language tags:", lang_rows["entrytag_id"].nunique())
print("\nDepth distribution (how specific are tags):")
print(lang_rows["depth"].value_counts().sort_index())

### a key thing: entries have more than one ###
pd.set_option('display.max_colwidth', None)
lang_rows.head(2)

''' entry id 23
Sino Tibetan (2 levels)
Old Chinese (4 levels) --> including two above.
'''

### for each entry ID find longest path + check whether shorter ones are a subset ###
id_re = re.compile(r"\[(\d+)\]")

def path_ids(p):
    # e.g. "Language[1747]->Sino-Tibetan[1946]" -> [1747, 1946]
    return [int(x) for x in id_re.findall(str(p))]

def is_prefix(short, long):
    # "subset" in this context = prefix along the lineage
    if len(short) > len(long):
        return False
    return short == long[:len(short)]

# 1) unique paths per entry (avoid duplicates)
df = (
    lang_rows.loc[:, ["entry_id", "entrytag_path"]]
    .dropna()
    .drop_duplicates()
    .copy()
)

df["ids"] = df["entrytag_path"].map(path_ids)
df["depth"] = df["ids"].map(len)

# sanity: every language path should have at least 1 id
assert (df["depth"] > 0).all()

# 2) for each entry_id, find the (a) longest path
longest = (
    df.sort_values(["entry_id", "depth"], ascending=[True, False])
      .groupby("entry_id", as_index=False)
      .head(1)
      .rename(columns={"entrytag_path": "longest_path", "ids": "longest_ids", "depth": "longest_depth"})
      .loc[:, ["entry_id", "longest_path", "longest_ids", "longest_depth"]]
)

longest # this is cool, huge variation (some only down to n=2)
longest.groupby('longest_depth').size().reset_index(name='count').sort_values('longest_depth')

# optional sanity: detect ties for "longest"
ties = (
    df.merge(longest[["entry_id", "longest_depth"]], on="entry_id", how="left")
      .query("depth == longest_depth")
      .groupby("entry_id").size()
)
n_ties = (ties > 1).sum()
print("entries with >1 longest path (tie):", n_ties)

'''
Okay so we do have branching for quite a few of these:
For instance Entry ID 2614.
Has at least two different n=7 depth paths.
'''

# 3) check: is every other path a prefix of the longest path?
df2 = df.merge(longest, on="entry_id", how="left")

df2["is_prefix_of_longest"] = df2.apply(
    lambda r: is_prefix(r["ids"], r["longest_ids"]),
    axis=1
)

# per entry: all paths contained?
summary = (
    df2.groupby("entry_id")["is_prefix_of_longest"]
       .all()
       .rename("all_contained")
       .reset_index()
)

print(summary["all_contained"].value_counts())

# 4) list entries that fail + print a few examples
bad_entries = summary.loc[~summary["all_contained"], "entry_id"].tolist()
print("n entries with non-contained paths:", len(bad_entries))

'''
More than half of all entries have brancing.
So we for sure need to look into this.
'''

### what if we just take the most overall level (n=2) ###
lang_two = lang_rows[lang_rows['depth']==2]
lang_two["entry_id"].nunique() # 747
lang_two.groupby("entry_id").size().reset_index(name='count').sort_values('count')

### okay even at this top level we do have branching ###
lang_two[lang_two['entry_id']==337]

'''
These cases I am not 100% sure how to handle.
So even the simplest case is tricky.
We could use something like "the most common tag at this level".
But maybe also possible to just use all of the branches.
'''

