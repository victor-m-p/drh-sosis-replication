"""
vmp 2023-04-04
"""

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

""" 
Take out relevant columns.

answer_value coded as: 
1: Ye
0: No
-1: Field doesn't know / I don't know (we recode this to np.nan)
"""

# Load data
data = pd.read_csv("../data/raw/answerset.csv")

# Define the questions to investigate and create mapping dataframe
question_coding = {
    "Are extra-ritual in-group markers present:": "extra-ritual in-group markers",
    "Are other religious groups in cultural contact with target religion:": "cultural contact",
    "Is there violent conflict (within sample region):": "violent internal",
    "Is there violent conflict (with groups outside the sample region):": "violent external",
    "Does membership in this religious group require permanent scarring or painful bodily alterations:": "permanent scarring",
    "Tattoos/scarification:": "tattoos/scarification",  # sub of extra-ritual in-group markers
    "Circumcision:": "circumcision",  # sub of extra-ritual in-group markers
    "Food taboos:": "food taboos",  # sub of extra-ritual in-group markers
    "Hair:": "hair",  # sub of extra-ritual in-group markers
    "Dress:": "dress",  # sub of extra-ritual in-group markers
    "Ornaments:": "ornaments",  # sub of extra-ritual in-group markers
    "Archaic ritual language:": "archaic ritual language",
}

# Take out the relevant columns
answers = data[
    [
        "poll_name",
        "entry_id",
        "question_id",
        "question_name",
        "parent_question_id",
        "answer_value",
    ]
].drop_duplicates()

# Subset the data to only include the questions of interest
answers_subset = answers[answers["question_name"].isin(question_coding.keys())]
answers_subset["question_short"] = answers_subset["question_name"].map(question_coding)

# Make sure that we are only working with groups
answers_subset = answers_subset[answers_subset["poll_name"].str.contains("Group")]

# Merge with questionrelation to get related names
questionrelations = pd.read_csv("../data/raw/questionrelation.csv")

# Handle this for Question ID
answers_subset = answers_subset.merge(
    questionrelations, on=["question_id", "poll_name"], how="inner"
)
answers_subset = answers_subset.drop(columns=["question_id"])
answers_subset = answers_subset.rename(columns={"related_question_id": "question_id"})

# Handle this for Parent Question ID
answers_subset["parent_question_id"] = (
    answers_subset["parent_question_id"].fillna(0).astype(int)
)
parent_question_mapping = (
    questionrelations.set_index("question_id")["related_question_id"].dropna().to_dict()
)
answers_subset["parent_question_id"] = answers_subset["parent_question_id"].replace(
    parent_question_mapping
)

### only keep answers that are 0 (no) or 1 (yes) ###
answers_subset = answers_subset[answers_subset["answer_value"].isin([0, 1])]

### remove inconsistent answers ###
# Identify inconsistent answers by checking if more than one exists for each (entry_id, question_id) group
answers_inconsistent = answers_subset.groupby(["entry_id", "question_id"]).size()
answers_inconsistent = answers_inconsistent[answers_inconsistent > 1].reset_index()[
    ["entry_id", "question_id"]
]

# Create a mapping from parent questions to child questions
parent_child_mapping = answers_subset[
    ["question_id", "parent_question_id"]
].drop_duplicates()
parent_child_mapping = parent_child_mapping[
    parent_child_mapping["parent_question_id"] != 0
]
parent_child_mapping = parent_child_mapping.rename(
    columns={"question_id": "child_question_id", "parent_question_id": "question_id"}
)

# Find all child questions for the inconsistent questions
affected_children = parent_child_mapping.merge(
    answers_inconsistent, on="question_id", how="inner"
)
affected_children = affected_children[["entry_id", "child_question_id"]].rename(
    columns={"child_question_id": "question_id"}
)

# Concatenate the original inconsistent questions with their affected child questions
answers_affected = pd.concat(
    [answers_inconsistent, affected_children]
).drop_duplicates()

# Remove all affected questions (both inconsistent questions and their children)
answers_subset_filtered = answers_subset[
    ~answers_subset.set_index(["entry_id", "question_id"]).index.isin(
        answers_affected.set_index(["entry_id", "question_id"]).index
    )
]

### unique combinations ###
from helper_functions import unique_combinations

unique_columns = ["question_id", "question_short", "parent_question_id"]
question_entry_combinations = unique_combinations(
    df=answers_subset_filtered,
    unique_columns=unique_columns,
    entry_column="entry_id",
    question_column="question_id",
)
answers_complete = question_entry_combinations[
    ["entry_id", "question_id", "question_short", "parent_question_id", "answer_value"]
]

# add back question name
question_names = answers_subset_filtered[
    ["question_name", "question_id"]
].drop_duplicates()
answers_complete = answers_complete.merge(question_names, on="question_id", how="inner")

### infer no if parent is no ###
from helper_functions import fill_answers

answers_inferred = fill_answers(answers_complete)
answers_inferred[answers_inferred["answer_inferred"] == "Yes"]

# save the data
answers_inferred.to_csv("../data/preprocessed/answers_conflict.csv", index=False)
