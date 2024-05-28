import pandas as pd
from scipy.stats import chi2_contingency


def code_conflict(row):
    if row["violent external"] == 1 and row["violent internal"] == 0:
        return "External only"
    elif row["violent external"] == 0 and row["violent internal"] == 1:
        return "Internal only"
    elif row["violent external"] == 1 and row["violent internal"] == 1:
        return "Internal and external"
    else:
        return "No violent conflict"


def code_conflict_collapsed(row):
    if row["violent external"] == 1:
        return "External Violent Conflict"
    else:
        return "No External Violent Conflict"


def code_internal_conflict(row):
    if row["violent internal"] == 1 and row["violent external"] == 0:
        return "Internal only"
    if row["violent internal"] == 0:
        return "No internal"
    else:
        return "Internal and external"


def run_chi2_test(df, marker):
    marker_df = df[df["marker"] == marker]
    marker_df["value"] = marker_df["value"].astype(int)
    contingency_table = pd.crosstab(marker_df["value"], marker_df["conflict_type"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p
