"""
vmp 2023-06-25:
Prepare answers for modeling
"""

import pandas as pd

# load
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
question_names_short = [
    # dependent variable
    "violent_external",
    # independent variables
    "circumcision",
    "tattoos_scarification",
    "permanent_scarring",
    "extra_ritual_group_markers",
    "food_taboos",
    "hair",
    "dress",
    "ornaments",
]
answers_subset = answers[answers["question_short"].isin(question_names_short)]
answers_subset = answers_subset[
    ["entry_id", "question_id", "question_short", "answer_value"]
]
answers_wide = answers_subset.pivot_table(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()

# entry region and time
entry_data = pd.read_csv("../data/preprocessed/entries_clean.csv")
entry_data = entry_data[["entry_id", "world_region", "year_from"]]
answers_time_region = answers_wide.merge(entry_data, on="entry_id", how="inner")

from helper_functions import process_time_region

# loop over variables
independent_variables = [
    "circumcision",
    "tattoos_scarification",
    "permanent_scarring",
    "extra_ritual_group_markers",
    "food_taboos",
    "hair",
    "dress",
    "ornaments",
]

for iv in independent_variables:
    # process data
    data_selection = process_time_region(
        answers_time_region,
        "entry_id",
        "violent_external",
        iv,
        "year_from",
        "world_region",
    )
    # save data
    data_selection.to_csv("../data/mdl_input/" + iv + ".csv", index=False)
