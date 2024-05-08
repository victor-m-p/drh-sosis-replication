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
    "extra-ritual in-group markers",
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


def fit_logistic_pymc(data, predictor, outcome):
    data[predictor] = pd.Categorical(data[predictor])
    with pm.Model() as model:
        # I don't think all of this is necesary
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
circumcision_trace = fit_logistic_pymc(circumcision, "violent external", "circumcision")
circumcision_summary = az.summary(circumcision_trace, var_names="beta", hdi_prob=0.95)
circumcision_summary_pct = convert_summary(circumcision_summary)
circumcision_summary_diff = effect_difference(
    circumcision_trace, circumcision_summary_pct
)

# tattoos/scarification
tattoos = preprocess_data(
    answers_wide, "entry_id", "violent external", "tattoos/scarification"
)
tattoos_trace = fit_logistic_pymc(tattoos, "violent external", "tattoos/scarification")
tattoos_summary = az.summary(tattoos_trace, var_names="beta", hdi_prob=0.95)
tattoos_summary_pct = convert_summary(tattoos_summary)
tattoos_summary_diff = effect_difference(tattoos_trace, tattoos_summary_pct)

# permanent scarring
scarring = preprocess_data(
    answers_wide, "entry_id", "violent external", "permanent scarring"
)
scarring_trace = fit_logistic_pymc(scarring, "violent external", "permanent scarring")
scarring_summary = az.summary(scarring_trace, var_names=["beta"], hdi_prob=0.95)
scarring_summary_pct = convert_summary(scarring_summary)
scarring_summary_diff = effect_difference(scarring_trace, scarring_summary_pct)

# extra-ritual in-group markers
extra_ritual = preprocess_data(
    answers_wide, "entry_id", "violent external", "extra-ritual in-group markers"
)
extra_ritual_trace = fit_logistic_pymc(
    extra_ritual, "violent external", "extra-ritual in-group markers"
)
extra_ritual_summary = az.summary(extra_ritual_trace, var_names=["beta"], hdi_prob=0.95)
extra_ritual_summary_pct = convert_summary(extra_ritual_summary)
extra_ritual_summary_diff = effect_difference(
    extra_ritual_trace, extra_ritual_summary_pct
)

# collect and save and plot
circumcision_summary_diff["label"] = "Circumcision"
tattoos_summary_diff["label"] = "Tattoos/Scarification"
scarring_summary_diff["label"] = "Permanent Scarring"
extra_ritual_summary_diff["label"] = "Extra-ritual In-Group Markers"
summary_converted = pd.concat(
    [
        circumcision_summary_diff,
        tattoos_summary_diff,
        scarring_summary_diff,
        extra_ritual_summary_diff,
    ]
)
summary_converted.to_csv("../tables/external_conflict_summary.csv", index=False)

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
plt.savefig("../figures/external_conflict_bayesian.pdf", dpi=300, bbox_inches="tight")

### now control for time ###
entry_time = pd.read_csv("../data/preprocessed/entry_time.csv")
entry_time = entry_time[["entry_id", "year_from"]]
answers_wide_time = answers_wide.merge(entry_time, on="entry_id", how="inner")


def preprocess_data_time(data, id, predictor, outcome, time):
    data_subset = data[[id, predictor, outcome, time]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    data_subset[time] = data_subset[time].astype(int)
    data_subset["time_scaled"] = (
        data_subset[time] - data_subset[time].mean()
    ) / data_subset[time].std()
    return data_subset


def fit_time_logistic_pymc(data, predictor, outcome, time):
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        b_external = pm.Normal("b_external", mu=0, sigma=5)
        b_time = pm.Normal("b_time", mu=0, sigma=5)
        logit_p = intercept + b_external * data[predictor] + b_time * data[time]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome])
        trace = pm.sample(2000, return_inferencedata=True)
    return trace


### time is not significant in any of these models ###
### external warfare remains significant in all models when controlling ###
### not entirely sure how to report this ###
# circumcision
circumcision_time = preprocess_data_time(
    answers_wide_time, "entry_id", "violent external", "circumcision", "year_from"
)
circumcision_time_trace = fit_time_logistic_pymc(
    circumcision_time, "violent external", "circumcision", "time_scaled"
)
circumcision_summary = az.summary(
    circumcision_time_trace, var_names=["intercept", "b_external", "b_time"]
)

# tattoos/scarification
tattoos_time = preprocess_data_time(
    answers_wide_time,
    "entry_id",
    "violent external",
    "tattoos/scarification",
    "year_from",
)
tattoos_time_trace = fit_time_logistic_pymc(
    tattoos_time, "violent external", "tattoos/scarification", "time_scaled"
)
tattoos_summary = az.summary(
    tattoos_time_trace, var_names=["intercept", "b_external", "b_time"]
)

# permanent scarring
scarring_time = preprocess_data_time(
    answers_wide_time, "entry_id", "violent external", "permanent scarring", "year_from"
)
scarring_time_trace = fit_time_logistic_pymc(
    scarring_time, "violent external", "permanent scarring", "time_scaled"
)
scarring_summary = az.summary(
    scarring_time_trace, var_names=["intercept", "b_external", "b_time"]
)

# extra-ritual in-group markers
extra_ritual_time = preprocess_data_time(
    answers_wide_time,
    "entry_id",
    "violent external",
    "extra-ritual in-group markers",
    "year_from",
)
extra_ritual_time_trace = fit_time_logistic_pymc(
    extra_ritual_time,
    "violent external",
    "extra-ritual in-group markers",
    "time_scaled",
)
extra_ritual_summary = az.summary(
    extra_ritual_time_trace, var_names=["intercept", "b_external", "b_time"]
)

# gather
circumcision_summary["label"] = "Circumcision"
tattoos_summary["label"] = "Tattoos/Scarification"
scarring_summary["label"] = "Permanent Scarring"
extra_ritual_summary["label"] = "Extra-ritual In-Group Markers"
summary_time = pd.concat(
    [circumcision_summary, tattoos_summary, scarring_summary, extra_ritual_summary]
)
summary_time["outcome"] = summary_time.index
summary_time.to_csv("../tables/external_conflict_time_summary.csv", index=False)

### now control for region ###
entry_region = pd.read_csv("../data/preprocessed/entry_regions.csv")
entry_region = entry_region[["entry_id", "world_region"]]
answers_wide_region = answers_wide.merge(entry_region, on="entry_id", how="inner")


def preprocess_data_region(data, id, predictor, outcome, world_region):
    data_subset = data[[id, predictor, outcome, world_region]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    data_subset[world_region] = pd.Categorical(data_subset[world_region])
    return data_subset


def fit_region_logistic_pymc(data, outcome):
    with pm.Model() as model:
        # Hyperpriors for intercepts
        mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=5)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=5)

        # Hyperpriors for slopes
        mu_slope = pm.Normal("mu_slope", mu=0, sigma=5)
        sigma_slope = pm.HalfCauchy("sigma_slope", beta=5)

        # Non-centered random intercepts for regions
        offsets_intercepts = pm.Normal(
            "offsets_intercepts",
            mu=0,
            sigma=1,
            shape=len(data["world_region"].unique()),
        )
        intercepts = pm.Deterministic(
            "intercepts", mu_intercept + sigma_intercept * offsets_intercepts
        )

        # Non-centered random slopes for regions
        offsets_slopes = pm.Normal(
            "offsets_slopes", mu=0, sigma=1, shape=len(data["world_region"].unique())
        )
        slopes = pm.Deterministic("slopes", mu_slope + sigma_slope * offsets_slopes)

        # Data indexing for regions
        region_idx = pm.Data("region_idx", data["world_region"].cat.codes)

        # Logit function incorporating random slopes and intercepts
        logit_p = intercepts[region_idx] + slopes[region_idx] * data[
            "violent external"
        ].astype(float)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome].astype(int))

        # Sampling
        trace = pm.sample(
            draws=2000, tune=2000, target_accept=0.99, return_inferencedata=True
        )
        return trace


# extra-ritual in-group markers
extra_ritual_region = preprocess_data_region(
    answers_wide_region,
    "entry_id",
    "violent external",
    "extra-ritual in-group markers",
    "world_region",
)

extra_ritual_trace = fit_region_logistic_pymc(
    extra_ritual_region, "extra-ritual in-group markers"
)

extra_ritual_summary = az.summary(
    extra_ritual_trace,
    var_names=[
        "mu_intercept",
        "mu_slope",
        "intercepts",
        "slopes",
    ],
    hdi_prob=0.95,
)

# circumcision
circumcision_region = preprocess_data_region(
    answers_wide_region, "entry_id", "violent external", "circumcision", "world_region"
)
circumcision_region_trace = fit_region_logistic_pymc(
    circumcision_region, "circumcision"
)
circumcision_region_summary = az.summary(
    circumcision_region_trace,
    var_names=["mu_intercept", "mu_slope", "intercepts", "slopes"],
    hdi_prob=0.95,
)
circumcision_region_summary

# tattoos/scarification
tattoos_region = preprocess_data_region(
    answers_wide_region,
    "entry_id",
    "violent external",
    "tattoos/scarification",
    "world_region",
)
tattoos_region_trace = fit_region_logistic_pymc(tattoos_region, "tattoos/scarification")
tattoos_region_summary = az.summary(
    tattoos_region_trace,
    var_names=["mu_intercept", "mu_slope", "intercepts", "slopes"],
    hdi_prob=0.95,
)
tattoos_region_summary

# permanent scarring
scarring_region = preprocess_data_region(
    answers_wide_region,
    "entry_id",
    "violent external",
    "permanent scarring",
    "world_region",
)
scarring_region_trace = fit_region_logistic_pymc(scarring_region, "permanent scarring")
scarring_region_summary = az.summary(
    scarring_region_trace,
    var_names=["mu_intercept", "mu_slope", "intercepts", "slopes"],
    hdi_prob=0.95,
)
scarring_region_summary

#### testing one at a time is problematic ####
#### I do think that we need the shared model to work ####
