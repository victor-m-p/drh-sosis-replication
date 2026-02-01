# come back and document this
import pandas as pd
from itertools import product


def unique_combinations(
    df: pd.DataFrame, unique_columns: list, entry_column: str, question_column: str
) -> pd.DataFrame:
    """Fill in all possible combinations of questions and entries.

    Args:
        df (pd.DataFrame): dataframe with relevant columns
        unique_columns (list): question id columns
        entry_column (str): name of the entry column ("entry_id")
        question_column (str): name of the question column ("question_id")

    Returns:
        pd.DataFrame: _description_
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

def fill_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Infer "No" answers for children based on "No" answers for parents.

    Args:
        df (pd.DataFrame): DataFrame with columns "entry_id", "question_id", "parent_question_id", "answer_value"

    Returns:
        pd.DataFrame: Returns the DataFrame with a new column "answer_inferred" that indicates if the answer was inferred.
    """
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

def process_time_region(
    df: pd.DataFrame,
    entry_col: str,
    predictor_col: str,
    outcome_col: str,
    year_scaled_col: str,   # <- renamed for clarity
    region_col: str,
) -> pd.DataFrame:
    df_subset = df[[entry_col, predictor_col, outcome_col, year_scaled_col, region_col]]
    df_subset = df_subset.dropna()
    df_subset[predictor_col] = df_subset[predictor_col].astype(int)
    df_subset[outcome_col] = df_subset[outcome_col].astype(int)
    df_subset = df_subset.rename(columns={year_scaled_col: "year_scaled"})
    df_subset[region_col] = pd.Categorical(df_subset[region_col])
    return df_subset
