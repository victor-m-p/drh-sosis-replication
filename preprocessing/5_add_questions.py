import pandas as pd 
import numpy as np 

# only take entries in main analysis:
answerset_main = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = answerset_main['entry_id'].unique()

# load the general entrydata 
answerset = pd.read_csv("../data/raw/answerset.csv")
answerset[answerset['question_name']=="The society to which the religious group belongs is best characterized as (please choose one):"]
answerset = answerset[[
    "poll_name",
    "entry_id",
    "question_id",
    "question_name",
    "answer_value",
    "answer", # adding because of the states.
]].drop_duplicates()
answerset = answerset[answerset['entry_id'].isin(entry_ids)]

# All of the questions under "Enforcement" and "Warfare"
# NB: names better because they should persist across the polls
# The question IDs are not the same 
question_coding = {
    # enforcement
    'Does the religious group in question provide an institutionalized police force:': 'police_force_own',
    'Do the group’s adherents interact with an institutionalized police force provided by an institution(s) other than the religious group in question:': 'police_force_other', 
    'Does the religious group in question provide institutionalized judges:': 'judges_own',
    'Do the group’s adherents interact with an institutionalized judicial system provided by an an institution(s) other than the religious group in question:': 'judges_other', 
    'Does the religious group in question enforce institutionalized punishment:': 'punish_own',
    'Are the group’s adherents subject to institutionalized punishment enforced by an institution(s) other than the religious group in question:': 'punish_other',
    'Does the religious group in question have a formal legal code:': 'legal_code_own',
    'Are the group’s adherents subject to a formal legal code provided by institution(s) other than the religious group in question:': 'legal_code_other',
    # warfare 
    'Does religious group in question possess an institutionalized military:': 'military_possess',
    'Do the group’s adherents participate in an institutionalized military provided by institution(s) other than the religious group in question:': 'military_participate',
    'Are the group’s adherents protected by or subject to an institutionalized military provided by an institution(s) other than the religious group in question:': 'military_protected',
    # additional ones on groups 
    "The society to which the religious group belongs is best characterized as (please choose one):": "society_type"
}

answers_subset = answerset[answerset["question_name"].isin(question_coding.keys())]
answers_subset["question_short"] = answers_subset["question_name"].map(question_coding)

# Merge with questionrelation to get related names
questionrelations = pd.read_csv("../data/raw/questionrelation.csv")
answers_subset = answers_subset.merge(
    questionrelations, on=["question_id", "poll_name"], how="inner"
)
answers_subset = answers_subset.drop(columns=["question_id"])
answers_subset = answers_subset.rename(columns={"related_question_id": "question_id"})

### need to handle the society question separately / differently ###
# answer_value 1-5 maps to the same category across all language translations of the
# answer text (checked against raw data), so we can safely collapse on answer_value.
society_labels = {
    1: "A band",
    2: "A tribe",
    3: "A chiefdom",
    4: "A state",
    5: "An empire",
}
answers_society = answers_subset[answers_subset['question_short'] == 'society_type'].copy()

# drop "field/I don't know" (-1) and "other" (0)
answers_society = answers_society[answers_society['answer_value'].isin(society_labels.keys())]
answers_society['answer'] = answers_society['answer_value'].map(society_labels)

'''
9 entries have more than one distinct society_type code (different coders/poll
versions disagreed on the classification). We are mainly interested in the split
between State/Empire vs. other, so we resolve 8 of these 9 by hand instead of
dropping them outright:

- A state <-> An empire (7 entries: 492, 688, 1044, 1231, 1426, 1903, 2365)
  Adjacent categories on the ordinal scale, and by far the most common
  disagreement (the state/empire boundary is the fuzziest one). Since we care
  about state-level vs. non-state societies, we resolve these to "A state".
- A chiefdom <-> A tribe (1 entry: 732)
  Also adjacent categories; resolved to "A chiefdom".
- A chiefdom <-> An empire (1 entry: 2240)
  A two-step jump on the ordinal scale rather than a borderline judgment call, so
  we drop this entry for now, pending discussion with collaborators.
'''

state_empire_resolve = [492, 688, 1044, 1231, 1426, 1903, 2365]
chiefdom_tribe_resolve = [732]
society_drop = [2240]

answers_society = answers_society[~answers_society['entry_id'].isin(society_drop)]
answers_society.loc[answers_society['entry_id'].isin(state_empire_resolve), 'answer'] = 'A state'
answers_society.loc[answers_society['entry_id'].isin(chiefdom_tribe_resolve), 'answer'] = 'A chiefdom'

# drop any remaining entries with more than one distinct society_type code
society_inconsistent = answers_society.groupby("entry_id")["answer"].nunique()
society_inconsistent = society_inconsistent[society_inconsistent > 1].index
answers_society = answers_society[~answers_society['entry_id'].isin(society_inconsistent)]

answers_society_wide = answers_society.drop_duplicates(subset=["entry_id"])[
    ["entry_id", "answer"]
].rename(columns={"answer": "society_type"})

# binary split: state/empire (1) vs. band/tribe/chiefdom (0)
answers_society_wide["state"] = answers_society_wide["society_type"].isin(
    ["A state", "An empire"]
).astype(int)

# remaining questions are simple binary yes/no, handled together
answers_subset = answers_subset[answers_subset['question_short'] != 'society_type']

# only keep answers that are 0 (no) or 1 (yes)
answers_subset = answers_subset[answers_subset["answer_value"].isin([0, 1])]

# Identify inconsistent answers by checking if more than one exists for each (entry_id, question_id) group
answers_inconsistent = answers_subset.groupby(["entry_id", "question_id"]).size()
answers_inconsistent = answers_inconsistent[answers_inconsistent > 1].reset_index()[
    ["entry_id", "question_id"]
] # n=0 so these two lines do nothing.

# merge wide and combine with original data
answers_wide = answers_subset.pivot_table(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()
answers_wide = answers_wide.merge(answers_society_wide, on="entry_id", how="outer")

# merge with our existing data (should be left here, fill NAN.)
answerset_large = answerset_main.merge(answers_wide, on = 'entry_id', how = 'left')
answerset_large.to_csv("../data/preprocessed/answerset_large.csv", index=False)
