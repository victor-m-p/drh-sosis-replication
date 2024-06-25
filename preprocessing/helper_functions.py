# come back and document this
import pandas as pd
from itertools import product


def check_data(df):
    n_rows = len(df)
    unique_entries = df["entry_id"].nunique()
    answer_distribution = df.groupby("answer_value", dropna=False).size()
    print(f"Number of rows: {n_rows}")
    print(f"Unique entries: {unique_entries}")
    print(f"Answer distribution: {answer_distribution}")


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


def fill_answers(df):
    df["answer_inferred"] = "No"
    for num, row in df.iterrows():
        # for children
        if row["parent_question_id"] != 0 and pd.isna(row["answer_value"]):
            parent = df[
                (df["entry_id"] == row["entry_id"])
                & (df["question_id"] == row["parent_question_id"])
            ]["answer_value"]

            # Check if there is exactly one element and if it's not NaN
            if len(parent) == 1 and not pd.isna(parent.iloc[0]):
                # Check if the value is 0
                if parent.iloc[0] == 0:
                    df.loc[num, "answer_value"] = 0
                    df.loc[num, "answer_inferred"] = "Yes"
    return df


def process_time_region(data, id, predictor, outcome, time, region):
    data_subset = data[[id, predictor, outcome, time, region]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    data_subset["year_scaled"] = (
        data_subset[time] - data_subset[time].mean()
    ) / data_subset[time].std()
    data_subset[region] = pd.Categorical(data_subset[region])
    return data_subset
