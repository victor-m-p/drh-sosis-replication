import pandas as pd

# load the data dump
data_raw = pd.read_csv("../data/raw/data_dump.csv")

# select relevant columns
data_raw = data_raw[
    [
        "poll",
        "question_id",
        "question_name",
        "parent_question_id",
        "entry_id",
        "entry_name",
        "answer",
        "value",
        "year_from",
        "year_to",
    ]
]

# save the data
data_raw.to_csv("../data/raw/raw_data.csv", index=False)
