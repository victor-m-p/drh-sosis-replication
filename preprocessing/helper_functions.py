# come back and document this

import pandas as pd
from itertools import product
import numpy as np


def check_data(df):
    n_rows = len(df)
    unique_entries = df["entry_id"].nunique()
    answer_distribution = df.groupby("answer_value", dropna=False).size()
    print(f"Number of rows: {n_rows}")
    print(f"Unique entries: {unique_entries}")
    print(f"Answer distribution: {answer_distribution}")


def map_question_relation(
    question_id_poll: pd.DataFrame, question_relation: pd.DataFrame, base_poll
):
    """
    Function assumes:
    1. question exists in both related_questions (as either question_id or related_question_id)
    2. question has a relation to a "Religious Group (v6)" question.
    3. the relation exists as a row in the question_relation table (in either direction)
    4. if this is not the case the question will be removed.

    Examples of questions that we remove (because they do not have a relation to group poll v6):

    Loop within Religious Place for instance:
    question_id: 5232, 6336, 5659, 6337, 5660, 5233

    Loop within Religious Text and Religious Place:
    question_id: 8411, 5637, 6061, 6738, 7571
    """

    # merge left
    merge_left = question_id_poll.merge(
        question_relation, on="question_id", how="inner"
    )
    # rename
    question_relation = question_relation.rename(
        columns={
            "question_id": "related_question_id",
            "related_question_id": "question_id",
        }
    )
    # merge right
    merge_right = question_id_poll.merge(
        question_relation, on="question_id", how="inner"
    )
    # concat
    df = pd.concat([merge_left, merge_right])
    # filter
    df = df[df["poll"] == base_poll]
    # remove poll
    df = df.drop(columns="poll")
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # insert missing self-links
    unique_question_ids = df["question_id"].unique()
    unique_related_question_ids = df["related_question_id"].unique()
    missing_related_ids = [
        qid for qid in unique_question_ids if qid not in unique_related_question_ids
    ]
    new_rows = pd.DataFrame(
        {"question_id": missing_related_ids, "related_question_id": missing_related_ids}
    )
    if len(new_rows) > 0:
        df = pd.concat([df, new_rows], ignore_index=True)
    # switch labels
    df = df.rename(
        columns={
            "question_id": "related_question_id",
            "related_question_id": "question_id",
        }
    )
    # now add back in the labels
    df = df.merge(question_id_poll, on="question_id", how="inner")
    # sort by question id
    df = df.sort_values(by=["question_id"])
    # tests
    assert df["related_question_id"].nunique() <= df["question_id"].nunique()

    return df


def unique_combinations(
    df: pd.DataFrame, unique_columns: list, entry_column: str, question_column: str
) -> pd.DataFrame:
    """
    Unique combinations of entries and questions.
    Fills in missing combinations with NaNs.
    """
    combinations_questions = df[unique_columns].drop_duplicates()
    entry_ids = df[entry_column].unique()
    question_ids = df[question_column].unique()
    product_questions_entries = pd.DataFrame(
        product(entry_ids, question_ids), columns=[entry_column, question_column]
    )
    combinations_filled = product_questions_entries.merge(
        combinations_questions, on=question_column, how="inner"
    )
    df = combinations_filled.merge(df, on=[entry_column] + unique_columns, how="left")

    # df >= product because it should have all combinations and some entries
    # will have multiple answers for the same question (or other duplication)
    assert len(df) >= len(entry_ids) * len(question_ids)
    return df
