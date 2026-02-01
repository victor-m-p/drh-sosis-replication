"""
vmp 2026-01-31 (update.)
Prepare answers for modeling. 
Combine entry data (year, region) and processed answers.
Save csv to mdl_input folder for each markers (dependent variable). 
"""

import pandas as pd

# load
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
question_names_short = [
    "violent_external",
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
answers_subset = answers_subset[["entry_id", "question_id", "question_short", "answer_value"]]

answers_wide = answers_subset.pivot_table(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()

# entry region and time
entry_data = pd.read_csv("../data/preprocessed/entries_clean.csv")
entry_data = entry_data[["entry_id", "world_region", "year_from"]]
answers_time_region = answers_wide.merge(entry_data, on="entry_id", how="inner")

# global scaling of year computed once
year_col = "year_from"
year_mean = answers_time_region[year_col].dropna().mean()
year_sd = answers_time_region[year_col].dropna().std()
answers_time_region["year_scaled"] = (answers_time_region[year_col] - year_mean) / year_sd

# information
answers_time_region["entry_id"].nunique() # n=771
answers_time_region[~answers_time_region["violent_external"].isna()]["entry_id"].nunique() # 551

# save scaling parameters for documentation
pd.DataFrame(
    {"year_col": [year_col], "year_mean": [year_mean], "year_sd": [year_sd]}
).to_csv("../data/documentation/year_scaling.csv", index=False)

from helper_functions import process_time_region

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
    data_selection = process_time_region(
        answers_time_region,
        "entry_id",
        "violent_external",
        iv,
        "year_scaled",        # passing year_scaled now
        "world_region",
    )
    data_selection.to_csv("../data/mdl_input/" + iv + ".csv", index=False)
