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


def extract_param(d, param):
    # extract
    estimate = d[d["parameter"] == param]["Estimate"].values[0]
    lower = d[d["parameter"] == param]["l-95% CI"].values[0]
    upper = d[d["parameter"] == param]["u-95% CI"].values[0]
    # round
    estimate = np.round(estimate, 2)
    lower = np.round(lower, 2)
    upper = np.round(upper, 2)
    # string
    param_string = f"{estimate} [{lower}; {upper}]"
    return param_string


output_files = os.listdir("../data/mdl_output")
logit_files = [i for i in output_files if "summary" in i]
percent_files = [i for i in output_files if "results" in i]

summary_list = []
for filename, label in convert_labels.items():
    # load files
    data_logit = pd.read_csv("../data/mdl_output/" + filename + "_summary.csv")
    data_percent = pd.read_csv("../data/mdl_output/" + filename + "_results.csv")
    # extract params
    estimate_logit = extract_param(data_logit, "violent_external")
    year_logit = extract_param(data_logit, "year_scaled")
    intercept_logit = extract_param(data_logit, "Intercept")
    estimate_percent = extract_param(data_percent, "effect")
    # collect
    summary_row = pd.DataFrame(
        {
            "Outcome": label,
            "Intercept": intercept_logit,
            "E. V. Conflict": estimate_logit,
            "E. V. Conflict (%)": estimate_percent,
            "Start Year": year_logit,
        },
        index=[0],
    )
    summary_list.append(summary_row)
summary_df = pd.concat(summary_list)

# rename columns and save as latex
summary_df.to_latex("../tables/brms_table.tex", index=False)
