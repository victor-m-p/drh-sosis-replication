import pandas as pd

pd.set_option("display.max_colwidth", None)

# load data
answerset = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# create parent_question_name column
question_names = answerset[["question_name", "question_id"]].drop_duplicates()
question_names_dict = dict(
    zip(question_names["question_id"], question_names["question_name"])
)
answerset["parent_question_name"] = answerset["parent_question_id"].map(
    question_names_dict
)

# take out the relevant columns
question_names = answerset[["question_name", "parent_question_name"]].drop_duplicates()
question_names = question_names.sort_values("parent_question_name").reset_index(
    drop=True
)
question_names = question_names.rename(
    columns={
        "question_name": "Question Name",
        "parent_question_name": "Parent Question Name",
    },
)
question_names.head(15)

# write to latex table
question_names.to_latex(
    "../tables/question_names.tex",
    index=False,
)
