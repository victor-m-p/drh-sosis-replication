import pandas as pd 
import numpy as np 

# only take entries in main analysis:
answerset_main = pd.read_csv("../data/preprocessed/answerset.csv")
entry_ids = answerset_main['entry_id'].unique()

# load the general entrydata 
answerset = pd.read_csv("../data/raw/answerset.csv")
answerset = answerset[[
    "poll_name",
    "entry_id",
    "question_id",
    "question_name",
    "answer_value"
]].drop_duplicates()
answerset = answerset[answerset['entry_id'].isin(entry_ids)]

# questionrelations


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

# merge with our existing data (should be left here, fill NAN.)
answerset_large = answerset_main.merge(answers_wide, on = 'entry_id', how = 'left')
answerset_large.to_csv("../data/preprocessed/answerset_large.csv", index=False)
