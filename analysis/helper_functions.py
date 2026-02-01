"""
VMP 2024-08-01
Helper functions for analysis.
"""

import pandas as pd
from scipy.stats import chi2_contingency


def code_external_conflict(row):
    """Helper function to code external conflict"""
    if row["violent_external"] == 1:
        return "External Violent Conflict"
    else:
        return "No External Violent Conflict"

def code_internal_conflict(row):
    """Helper function to code internal conflict"""
    if row["violent_internal"] == 1 and row["violent_external"] == 0:
        return "Internal Conflict only"
    if row["violent_internal"] == 0:
        return "No Internal Conflict"
    else:
        return "Internal and External"

def run_chi2_test(df, marker):
    """Helper function to run chi2 test"""
    marker_df = df[df["marker"] == marker]
    marker_df["value"] = marker_df["value"].astype(int)
    contingency_table = pd.crosstab(marker_df["value"], marker_df["conflict_type"])
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=False)
    return chi2, p
