"""
vmp 2026-07-01
Creates an overview table of the questions coded in 1_curate_data.py:
question name and parent question name.
"""

import os
import pandas as pd

os.makedirs("../tables", exist_ok=True)

# Same questions of interest as in 1_curate_data.py
questions_of_interest = [
    # independent variables
    "Are other religious groups in cultural contact with target religion:",
    "Is there violent conflict (within sample region):",
    "Is there violent conflict (with groups outside the sample region):",
    # dependent variables
    "Are extra-ritual in-group markers present:",
    "Does membership in this religious group require permanent scarring or painful bodily alterations:",
    # sub-questions of extra-ritual in-group markers
    "Tattoos/scarification:",
    "Circumcision:",
    "Food taboos:",
    "Hair:",
    "Dress:",
    "Ornaments:",
]

# Load data
data = pd.read_csv("../data/raw/answerset.csv")
questionrelations = pd.read_csv("../data/raw/questionrelation.csv")

answers = data[
    ["poll_name", "question_id", "question_name", "parent_question_id"]
].drop_duplicates()

# Subset to questions of interest, group polls only
answers_subset = answers[answers["question_name"].isin(questions_of_interest)]
answers_subset = answers_subset[answers_subset["poll_name"].str.contains("Group")]

# Harmonize question_id across poll versions (as in 1_curate_data.py)
answers_subset = answers_subset.merge(
    questionrelations, on=["question_id", "poll_name"], how="inner"
)
answers_subset = answers_subset.drop(columns=["question_id"]).rename(
    columns={"related_question_id": "question_id"}
)

# Harmonize parent_question_id the same way
answers_subset["parent_question_id"] = answers_subset["parent_question_id"].fillna(0).astype(int)
parent_id_mapping = (
    questionrelations.set_index("question_id")["related_question_id"].dropna().to_dict()
)
answers_subset["parent_question_id"] = answers_subset["parent_question_id"].replace(parent_id_mapping)

# One row per question
question_table = answers_subset[
    ["question_id", "question_name", "parent_question_id"]
].drop_duplicates()

# Look up parent question name via its harmonized question_id
question_name_lookup = question_table.set_index("question_id")["question_name"].to_dict()
question_table["parent_question_name"] = question_table["parent_question_id"].map(question_name_lookup)

question_table = question_table.rename(
    columns={"question_name": "Question Name", "parent_question_name": "Parent Question Name"}
)
question_table = question_table[["Question Name", "Parent Question Name"]]

# Drop the cultural-contact superquestion row; it only serves to impute data
# for the two violent-conflict variables and already appears as their parent.
question_table = question_table[
    question_table["Question Name"]
    != "Are other religious groups in cultural contact with target religion:"
]

# Custom top-level ordering; sub-questions of extra_ritual_group_markers
# are sorted alphabetically underneath it.
group_order = [
    "Does membership in this religious group require permanent scarring or painful bodily alterations:",
    "Are extra-ritual in-group markers present:",
    "Is there violent conflict (with groups outside the sample region):",
    "Is there violent conflict (within sample region):",
]
question_table["group"] = question_table["Parent Question Name"].fillna(question_table["Question Name"])
question_table["is_child"] = question_table["Question Name"] != question_table["group"]
question_table["group_rank"] = question_table["group"].map({name: i for i, name in enumerate(group_order)})

question_table = (
    question_table.sort_values(["group_rank", "is_child", "Question Name"])
    .drop(columns=["group", "is_child", "group_rank"])
    .reset_index(drop=True)
)

# Write LaTeX table (compact, wrapped columns for long question names)
lines = []
lines.append("\\begin{table}[ht]")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\renewcommand{\\arraystretch}{1.2}")
lines.append("\\caption{Overview of coded questions.}")
lines.append("\\begin{tabular}{p{6.5cm}p{6.5cm}}")
lines.append("\\toprule")
lines.append("Question Name & Parent Question Name \\\\")
lines.append("\\midrule")

for _, row in question_table.iterrows():
    parent = row["Parent Question Name"] if pd.notna(row["Parent Question Name"]) else "--"
    name = row["Question Name"]
    if parent == "Are extra-ritual in-group markers present:":
        name = f"\\hspace{{1em}}{name}"
    lines.append(f"{name} & {parent} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open("../tables/question_coding.tex", "w") as f:
    f.write("\n".join(lines))
