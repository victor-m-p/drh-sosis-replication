'''
Loads manual coding C.
Cleans the table by back-inferring values (e.g., ASJP/Glotto from tips).
Then removes columns that are not needed.
Writes a clean coding table.
'''

import pandas as pd

lang = pd.read_csv("../data/raw/manual_lang_C.csv")
ref = pd.read_csv("data/tip_to_glottocode.csv")  # tip_name -> asjp_id, glottocode, glottolog_name

# tidy the expert-entered columns
lang["tip_assigned"] = lang["tip_assigned"].str.strip()
lang["tip_confidence"] = lang["tip_confidence"].str.strip().replace({"medum": "medium"})

# fill in asjp_id / glottocode / glottolog_name from the assigned Jaeger tip (unique lookup)
lang = lang.drop(columns=["glottolog_name", "glottolog_id", "asjp_id"])
lang = lang.merge(ref, left_on="tip_assigned", right_on="tip_name", how="left")
lang = lang.rename(columns={"glottocode": "glottolog_id"}).drop(columns="tip_name")

lang = lang[["entry_id", "entry_name", "entrytag_name",
             "tip_assigned", "asjp_id", "glottolog_id", "glottolog_name",
             "tip_coder", "tip_confidence"]]

lang.to_csv("../data/preprocessed/language_master.csv", index=False)
