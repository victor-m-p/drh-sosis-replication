"""
External violent conflict vs. Extra-ritual in-group markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_functions import logistic_to_percent
import pymc as pm
import arviz as az
import numpy as np


# load
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")
answers.groupby(["question_id", "question_short"]).size()

# 1. circumcision, tattoos/scarification, and permanent scarring by external conflict
question_names_short = [
    "violent external",
    "circumcision",
    "tattoos/scarification",
    "permanent scarring",
]
answers_subset = answers[answers["question_short"].isin(question_names_short)]
answers_subset = answers_subset[
    ["entry_id", "question_id", "question_short", "answer_value"]
]
answers_wide = answers_subset.pivot_table(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()


# functions for this section
def preprocess_data(data, id, predictor, outcome):
    data_subset = data[[id, predictor, outcome]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    return data_subset


def fit_logistic_model(data, predictor, outcome):
    data[predictor] = pd.Categorical(data[predictor])
    with pm.Model() as model:
        # Setting up a separate intercept for each region
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=5,
            shape=len(data[predictor].cat.categories),
        )

        # Get the index for each region from the categorical variable
        external_idx = pm.ConstantData("external_idx", data[predictor].cat.codes)

        # Probability of circumcision for each observation based on its region
        p = pm.Deterministic("p", pm.math.sigmoid(beta[external_idx]))

        # Likelihood
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome])

        # Sample from the posterior
        trace = pm.sample(2000, return_inferencedata=True)

    return trace


def convert_summary(summary):
    # convert mean effects
    summary["model_parameter"] = summary.index
    parameter_replacements = {"beta[0]": "No External", "beta[1]": "External"}
    summary["parameter"] = summary["model_parameter"].replace(parameter_replacements)

    # for each parameter get the values out
    summary["mean_pct"] = logistic_to_percent(summary["mean"])
    summary["hdi_2.5%_pct"] = logistic_to_percent(summary["hdi_2.5%"])
    summary["hdi_97.5%_pct"] = logistic_to_percent(summary["hdi_97.5%"])
    # summary["error_neg"] = summary["mean_pct"] - summary["hdi_2.5%_pct"]
    # summary["error_pos"] = summary["hdi_97.5%_pct"] - summary["mean_pct"]

    # select columns
    summary = summary[
        ["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"]
    ].reset_index(drop=True)
    return summary


def effect_difference(trace, summary_pct):
    beta_0_samples = trace.posterior["beta"].values[:, :, 0].flatten()
    beta_1_samples = trace.posterior["beta"].values[:, :, 1].flatten()

    beta_0_samples_pct = logistic_to_percent(beta_0_samples)
    beta_1_samples_pct = logistic_to_percent(beta_1_samples)

    effect_difference_samples = beta_1_samples_pct - beta_0_samples_pct

    hdi = az.hdi(effect_difference_samples, hdi_prob=0.95)
    mean_effect_difference = np.mean(effect_difference_samples)

    dataframe_difference = pd.DataFrame(
        columns=["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"],
        data=[("Yes - No", mean_effect_difference, hdi[0], hdi[1])],
    )

    summary_pct = pd.concat([summary_pct, dataframe_difference])
    summary_pct["error_neg"] = summary_pct["mean_pct"] - summary_pct["hdi_2.5%_pct"]
    summary_pct["error_pos"] = summary_pct["hdi_97.5%_pct"] - summary_pct["mean_pct"]
    return summary_pct


# circumcision
circumcision = preprocess_data(
    answers_wide, "entry_id", "violent external", "circumcision"
)
circumcision_trace = fit_logistic_model(
    circumcision, "violent external", "circumcision"
)
circumcision_summary = az.summary(circumcision_trace, var_names="beta", hdi_prob=0.95)
circumcision_summary_pct = convert_summary(circumcision_summary)
circumcision_summary_diff = effect_difference(
    circumcision_trace, circumcision_summary_pct
)

# tattoos/scarification
tattoos = preprocess_data(
    answers_wide, "entry_id", "violent external", "tattoos/scarification"
)
tattoos_trace = fit_logistic_model(tattoos, "violent external", "tattoos/scarification")
tattoos_summary = az.summary(tattoos_trace, var_names="beta", hdi_prob=0.95)
tattoos_summary_pct = convert_summary(tattoos_summary)
tattoos_summary_diff = effect_difference(tattoos_trace, tattoos_summary_pct)

# permanent scarring
scarring = preprocess_data(
    answers_wide, "entry_id", "violent external", "permanent scarring"
)
scarring_trace = fit_logistic_model(scarring, "violent external", "permanent scarring")
scarring_summary = az.summary(scarring_trace, var_names=["beta"], hdi_prob=0.95)
scarring_summary_pct = convert_summary(scarring_summary)
scarring_summary_diff = effect_difference(scarring_trace, scarring_summary_pct)

# collect and save and plot
circumcision_summary_diff["label"] = "Circumcision"
tattoos_summary_diff["label"] = "Tattoos/Scarification"
scarring_summary_diff["label"] = "Permanent Scarring"
summary_converted = pd.concat(
    [
        circumcision_summary_diff,
        tattoos_summary_diff,
        scarring_summary_diff,
    ]
)

# Plot configuration
fig, ax1 = plt.subplots(figsize=(8, 4))

# Position each effect with a slight offset within each group
spacing = 0.4  # space between effects in the same group
group_spacing = 3  # space between different groups
summary_converted["y_pos"] = summary_converted.apply(
    lambda row: list(summary_converted["label"].unique()).index(row["label"])
    * group_spacing
    + list(summary_converted["parameter"].unique()).index(row["parameter"]) * spacing,
    axis=1,
)

# Main plot
for label, group in summary_converted.groupby("label"):
    ax1.errorbar(
        group["mean_pct"],
        group["y_pos"],
        xerr=[group["error_neg"].values, group["error_pos"].values],
        fmt="o",
        label=label,
        color="black",  # colors[label],
        capsize=5,
    )

# Set primary y-axis
primary_labels = [label for label in summary_converted["label"].unique()]
ax1.set_yticks([i * group_spacing + spacing for i in range(len(primary_labels))])
ax1.set_yticklabels(primary_labels)
ax1.set_ylabel("")

# Create secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
secondary_labels = ["No External", "External", "Yes - No"] * len(
    summary_converted["label"].unique()
)
ax2.set_yticks([i for i in summary_converted["y_pos"]])
ax2.set_yticklabels(secondary_labels)

plt.xlabel("Mean Percentage")
plt.title("Violent External Conflict")
plt.grid(True, axis="x")
plt.show()
