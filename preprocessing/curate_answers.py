"""
vmp 2023-04-04
"""

import pandas as pd
from helper_functions import check_data

""" 
Take out relevant columns.

answer_value coded as: 
1: Ye
0: No
-1: Field doesn't know / I don't know (we recode this to np.nan)
"""

# load data
data = pd.read_csv("../data/raw/answersets.csv")
data = data.rename(columns={"value": "answer_value"})

# take out the relevant columns
answers = data[
    [
        "poll",
        "entry_id",
        "question_id",
        "question_name",
        "parent_question_id",
        "answer_value",
    ]
].drop_duplicates()

# fillna with placeholder value to convert to int
answers["parent_question_id"] = answers["parent_question_id"].fillna(0).astype(int)
check_data(answers)

""" 
notes:
"""

question_relation = pd.read_csv("../data/raw/question_relation.csv")
question_id_poll = answers[["question_id", "poll"]].drop_duplicates()

from helper_functions import map_question_relation

related_questions = map_question_relation(
    question_id_poll, question_relation, "Religious Group (v6)"
)

# sanity check
assert len(related_questions) == related_questions["question_id"].nunique()

# merge to get related questions
answers = answers.merge(related_questions, on=["question_id", "poll"], how="inner")

check_data(answers)

""" 
notes:
"""

# could also include the additional question we discussed with Ted.
question_coding = {
    "Are extra-ritual in-group markers present:": "extra-ritual in-group markers",
    "Are other religious groups in cultural contact with target religion:": "cultural contact",
    "Is there violent conflict (within sample region):": "violent internal",
    "Is there violent conflict (with groups outside the sample region):": "violent external",
    "Does membership in this religious group require permanent scarring or painful bodily alterations:": "permanent scarring",
    "Tattoos/scarification:": "tattoos/scarification",
    "Circumcision:": "circumcision",
}

# from constants import question_coding
short_question_names = pd.DataFrame(
    list(question_coding.items()), columns=["question_name", "question_short"]
)

# subset questions
answers_subset = answers.merge(short_question_names, on="question_name", how="inner")
answers_subset.groupby(["question_id", "question_name"]).size()

# now set related questions as the actual question ID that we use
answers_subset = answers_subset.drop(columns=["question_id"])
answers_subset = answers_subset.rename(columns={"related_question_id": "question_id"})

# now make sure that parent question id is also consistent
# we base this on the v6 poll
parent_ids = answers_subset[answers_subset["poll"] == "Religious Group (v6)"][
    ["question_id", "parent_question_id"]
].drop_duplicates()
answers_subset = answers_subset.drop(columns=["parent_question_id"])
answers_subset = answers_subset.merge(parent_ids, on="question_id", how="inner")

# okay these questions naturally only select group (v5, v6)
# and the formulation is the exact same
# so we do not have to filter here.

# now we first clean inconsistent answers
answers_inconsistent = (
    answers_subset.groupby(["entry_id", "question_name", "question_id"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
)
answers_inconsistent = answers_inconsistent[answers_inconsistent["n"] > 1]
answers_inconsistent = answers_inconsistent.sort_values(by=["entry_id", "question_id"])
answers_inconsistent

correct_answers = [
    [192, 5161, 1],  # yes and don't know
    [228, 4654, 1],  # yes (not clear why I have 2 answers here)
    [455, 5161, 1],  # yes
    [455, 5161, 0],  # no
    [572, 4658, 0],  # no
    [572, 4658, 1],  # yes
    [574, 4658, 0],  # no
    [574, 4658, 1],  # yes
    [574, 4659, 0],  # no
    [574, 4659, 1],  # yes
    [638, 4654, 1],  # yes (not clear why I have 2 answers here)
    [645, 5161, 1],  # yes
    [727, 4658, 1],  # yes
    [741, 5161, 1],  # yes
    [741, 5161, 0],  # no
    [775, 4658, 0],  # no
    [775, 4658, 1],  # yes
    [775, 4659, 0],  # no
    [775, 4659, 1],  # yes
    [893, 4658, -1],  # don't know
    [967, 4658, 1],  # yes
    [967, 4658, 0],  # no
    [1016, 4659, 1],  # yes
    [1016, 4659, 0],  # no
    [1038, 5161, 1],  # yes
    [1175, 4654, 1],  # yes
    [1268, 5162, 1],  # yes and don't know (weak yes)
    [1466, 4658, 1],  # yes
    [1466, 4658, 0],  # no
    [1466, 4659, 1],  # yes
    [1466, 4659, 0],  # no
    [1619, 4654, 1],  # yes
    [1621, 4654, 1],  # yes
    [1805, 4654, 1],  # yes (not clear why I have 2 answers here)
]

correct_answers = pd.DataFrame(
    correct_answers, columns=["entry_id", "question_id", "answer_value"]
)

len(answers_subset)
merged_answers = pd.merge(
    answers_subset,
    correct_answers,
    on=["entry_id", "question_id"],
    how="left",
    suffixes=("", "_small"),
)

df_filtered = merged_answers[
    (merged_answers["answer_value"] == merged_answers["answer_value_small"])
    | merged_answers["answer_value_small"].isnull()
]
assert len(df_filtered) <= len(answers_subset)
answers_subset = df_filtered.drop(columns=["answer_value_small"])
check_data(answers_subset)

""" 
notes:
"""

# rename to related
from helper_functions import unique_combinations

combination_list = []
for df_poll in answers_subset["poll"].unique():
    poll_subset = answers_subset[answers_subset["poll"] == df_poll]
    unique_columns = [
        "poll",
        "question_id",
        "parent_question_id",
        "question_name",
        "question_short",
    ]
    poll_combinations = unique_combinations(
        df=poll_subset,
        unique_columns=unique_columns,
        entry_column="entry_id",
        question_column="question_id",
    )
    combination_list.append(poll_combinations)
answers_subset = pd.concat(combination_list)

# recode -1 to np.nan
answers_subset["answer_value"] = answers_subset["answer_value"].replace(-1, pd.NA)
check_data(answers_subset)

#### infer "No" answers from parents ####
# first code questions into parents and children.
answers_subset.groupby(["question_id", "question_name"]).size()
question_level = [
    (4654, "parent"),
    (4658, "child"),
    (4659, "child"),
    (5161, "parent"),
    (5130, "parent"),
    (5162, "child"),
    (5163, "child"),
]
question_level = pd.DataFrame(question_level, columns=["question_id", "level"])
answers_subset = answers_subset.merge(question_level, on="question_id", how="inner")

# the hard part is what we do for sub-questions if parents are inconsistent
# I think for now we simply remove all entries with inconsistent data
# This removes n=9 entries.
n_questions = len(question_level)
answers_inconsistent = (
    answers_subset.groupby(["entry_id"]).size().reset_index(name="count")
)
answers_inconsistent = answers_inconsistent[answers_inconsistent["count"] > n_questions]
entries_inconsistent = answers_inconsistent["entry_id"].unique()
answers_subset = answers_subset[~answers_subset["entry_id"].isin(entries_inconsistent)]

# now we infer the "No" answers:
answers_subset["answer_inferred"] = "No"

for num, row in answers_subset.iterrows():
    if row["level"] == "child" and pd.isna(row["answer_value"]):
        parent = answers_subset[
            (answers_subset["entry_id"] == row["entry_id"])
            & (answers_subset["question_id"] == row["parent_question_id"])
        ]["answer_value"]

        # Check if there is exactly one element and if it's not NaN
        if len(parent) == 1 and not pd.isna(parent.iloc[0]):
            # Check if the value is 0
            if parent.iloc[0] == 0:
                answers_subset.loc[num, "answer_value"] = 0
                answers_subset.loc[num, "answer_inferred"] = "Yes"

# save the data
answers_subset.to_csv("../data/preprocessed/answers_conflict.csv", index=False)
