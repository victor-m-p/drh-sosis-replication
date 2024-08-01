"""
VMP 2024-07-31
We do not report these tables, but we do report these results (supplementary information).
We are using draws from the Bayesian analysis (brms_models_hraf.R) here.
"""

import numpy as np
import pandas as pd
import os

convert_labels = {
    "circumcision": "Circumcision",
    "dress": "Dress",
    "extra_ritual_group_markers": "Extra-Ritual In-Group Markers",
    "food_taboos": "Food Taboos",
    "hair": "Hair",
    "ornaments": "Ornaments",
    "permanent_scarring": "Permanent Scarring",
    "tattoos_scarification": "Tattoos or Scarification",
}


def extract_param(d, param, multiplier=1):
    # extract
    estimate = d[d["parameter"] == param]["Estimate"].values[0] * multiplier
    lower = d[d["parameter"] == param]["l-95% CI"].values[0] * multiplier
    upper = d[d["parameter"] == param]["u-95% CI"].values[0] * multiplier
    # round
    estimate = np.round(estimate, 2)
    lower = np.round(lower, 2)
    upper = np.round(upper, 2)
    # string
    param_string = f"{estimate} [{lower}; {upper}]"
    return param_string


### create table of all results ###
output_files = os.listdir("../data/mdl_output_hraf")
logit_files = [i for i in output_files if "summary" in i]

summary_list = []
for filename, label in convert_labels.items():
    # load files
    data_logit = pd.read_csv("../data/mdl_output_hraf/" + filename + "_summary.csv")
    # extract params
    estimate_logit = extract_param(data_logit, "violent_external")
    year_logit = extract_param(data_logit, "year_scaled")
    intercept_logit = extract_param(data_logit, "Intercept")
    # collect
    summary_row = pd.DataFrame(
        {
            "Outcome": label,
            "Intercept": intercept_logit,
            "External Violent Conflict": estimate_logit,
            "Start Year": year_logit,
        },
        index=[0],
    )
    summary_list.append(summary_row)
summary_df = pd.concat(summary_list)

# rename columns and save as latex
summary_df.to_latex("../tables/brms_table_hraf.tex", index=False)

### create table of important results ###
output_files = os.listdir("../data/mdl_output_hraf")
percent_files = [i for i in output_files if "results" in i]
hypothesis_files = [i for i in output_files if "hypothesis" in i]
logit_files = [i for i in output_files if "summary" in i]
summary_list = []
for filename, label in convert_labels.items():
    # load files
    data_percent = pd.read_csv("../data/mdl_output_hraf/" + filename + "_results.csv")
    data_hypothesis = pd.read_csv(
        "../data/mdl_output_hraf/" + filename + "_hypotheses.csv"
    )
    data_logit = pd.read_csv("../data/mdl_output_hraf/" + filename + "_summary.csv")
    data_hypothesis = data_hypothesis[
        data_hypothesis["Hypothesis"] == "(violent_external) > 0"
    ]
    # extract params
    estimate_percent = extract_param(data_percent, "effect", 100)
    estimate_logit = extract_param(data_logit, "violent_external")
    evidence_ratio = data_hypothesis["Evid.Ratio"].values[0]
    posterior_probability = data_hypothesis["Post.Prob"].values[0]
    # collect
    summary_row = pd.DataFrame(
        {
            "Outcome": label,
            "External Violent Conflict": estimate_logit,
            "External Violent Conflict (%)": estimate_percent,
            "Evid. Ratio": round(evidence_ratio, 2),
            "Post. Prob": round(posterior_probability, 2),
        },
        index=[0],
    )
    summary_list.append(summary_row)
summary_df = pd.concat(summary_list)
summary_df = summary_df.sort_values("Evid. Ratio", ascending=False)
summary_df.to_latex("../tables/brms_table_main_hraf.tex", index=False)
