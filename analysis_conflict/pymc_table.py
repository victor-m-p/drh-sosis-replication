import numpy as np
import pandas as pd
import os

convert_labels = {
    "archaic_language": "Archaic Language",
    "circumcision": "Circumcision",
    "dress": "Dress",
    "extra_ritual_markers": "Extra-Ritual In-Group Markers",
    "food_taboos": "Food Taboos",
    "hair": "Hair",
    "ornaments": "Ornaments",
    "permanent_scarring": "Permanent Scarring",
    "tattoos_scarification": "Tattoos/Scarification",
}


### for percent ###
def extract_beta_pct(d, n_round=2):
    # extract values
    beta = d[d["parameter"] == "Slope"]["mean_pct"].values[0]
    hdi_lower = d[d["parameter"] == "Slope"]["hdi_2.5%_pct"].values[0]
    hdi_upper = d[d["parameter"] == "Slope"]["hdi_97.5%_pct"].values[0]
    # round
    beta = np.round(beta, n_round)
    hdi_lower = np.round(hdi_lower, n_round)
    hdi_upper = np.round(hdi_upper, n_round)
    # string
    beta_string = f"{beta} [{hdi_lower}; {hdi_upper}]"
    return beta_string


summary_files_pct = os.listdir("../mdl_output/pct")
summary_files_pct = sorted(summary_files_pct)
summary_list = []
for i in summary_files_pct:
    d = pd.read_csv("../mdl_output/pct/" + i)
    x = extract_beta_pct(d)
    variable = i.split("_summary")[0]
    summary_list.append([variable, x])
summary_df = pd.DataFrame(summary_list, columns=["variable", "beta"])

# rename columns and save as latex
summary_df = summary_df.rename(columns={"variable": "Outcome", "beta": "Estimate"})
summary_df["Outcome"] = summary_df["Outcome"].map(convert_labels)
summary_df.to_latex("../tables/pymc_table_pct.tex", index=False)


### for raw ###
def extract_beta_raw(d, n_round=2):
    # extract values
    beta = d[d["variable"] == "mu_slope"]["mean"].values[0]
    hdi_lower = d[d["variable"] == "mu_slope"]["hdi_2.5%"].values[0]
    hdi_upper = d[d["variable"] == "mu_slope"]["hdi_97.5%"].values[0]
    # round
    beta = np.round(beta, n_round)
    hdi_lower = np.round(hdi_lower, n_round)
    hdi_upper = np.round(hdi_upper, n_round)
    # string
    beta_string = f"{beta} [{hdi_lower}; {hdi_upper}]"
    return beta_string


summary_files_raw = os.listdir("../mdl_output/raw")
summary_files_raw = sorted(summary_files_raw)
summary_list = []
for i in summary_files_raw:
    d = pd.read_csv("../mdl_output/raw/" + i)
    x = extract_beta_raw(d)
    variable = i.split("_summary")[0]
    summary_list.append([variable, x])
summary_df = pd.DataFrame(summary_list, columns=["variable", "beta"])

# rename columns and save as latex
summary_df = summary_df.rename(columns={"variable": "Outcome", "beta": "Estimate"})
summary_df["Outcome"] = summary_df["Outcome"].map(convert_labels)
summary_df.to_latex("../tables/pymc_table_raw.tex", index=False)
