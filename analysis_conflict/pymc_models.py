import pymc as pm
import pandas as pd
import arviz as az
from model_functions import logistic_to_percent
import numpy as np

# overall arguments
hdi_prob = 0.95
n_regions = 10


def fit_region_time_logistic_pymc(data, outcome):
    with pm.Model() as model:
        # Prior for time
        b_time = pm.Normal("b_time", mu=0, sigma=5)

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
        logit_p = (
            b_time * data["time_scaled"]
            + intercepts[region_idx]
            + slopes[region_idx] * data["violent external"].astype(float)
        )
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome].astype(int))

        # Sampling
        trace = pm.sample(
            draws=2000, tune=2000, target_accept=0.99, return_inferencedata=True
        )
        return trace


# load
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# 1. circumcision, tattoos/scarification, and permanent scarring by external conflict
question_names_short = [
    "violent external",
    "circumcision",
    "tattoos/scarification",
    "permanent scarring",
    "extra-ritual in-group markers",
    "archaic ritual language",
    "food taboos",
    "hair",
    "dress",
    "ornaments",
]
answers_subset = answers[answers["question_short"].isin(question_names_short)]
answers_subset = answers_subset[
    ["entry_id", "question_id", "question_short", "answer_value"]
]
answers_wide = answers_subset.pivot_table(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()


# entry region and time
entry_data = pd.read_csv("../data/preprocessed/entry_data.csv")
entry_data = entry_data[["entry_id", "world_region", "year_from"]]
answers_time_region = answers_wide.merge(entry_data, on="entry_id", how="inner")


def process_data_time_region(data, id, predictor, outcome, time, region):
    data_subset = data[[id, predictor, outcome, time, region]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    data_subset["time_scaled"] = (
        data_subset[time] - data_subset[time].mean()
    ) / data_subset[time].std()
    data_subset[region] = pd.Categorical(data_subset[region])
    return data_subset


### now 1 variable at a time ###
variable_grid = [
    ("circumcision", "circumcision"),
    ("tattoos/scarification", "tattoos_scarification"),
    ("permanent scarring", "permanent_scarring"),
    ("extra-ritual in-group markers", "extra_ritual_markers"),
    ("archaic ritual language", "archaic_language"),
    ("food taboos", "food_taboos"),
    ("hair", "hair"),
    ("dress", "dress"),
    ("ornaments", "ornaments"),
]

for outcome, label in variable_grid:
    # process data
    data_selection = process_data_time_region(
        answers_time_region,
        "entry_id",
        "violent external",
        outcome,
        "year_from",
        "world_region",
    )
    # fit model
    trace = fit_region_time_logistic_pymc(data_selection, outcome)
    # summary data
    summary = az.summary(
        trace,
        var_names=["b_time", "mu_intercept", "mu_slope", "intercepts", "slopes"],
        hdi_prob=hdi_prob,
    )
    summary["variable"] = summary.index

    # save basic summary
    summary.to_csv(f"../mdl_output/raw/{label}_summary.csv", index=False)

    ### convert to percent ###

    # mean and hdi for intercept
    mu_intercept_samples = trace.posterior["mu_intercept"].values.flatten()
    intercept_samples_pct = logistic_to_percent(mu_intercept_samples)
    intercept_hdi = az.hdi(intercept_samples_pct, hdi_prob=hdi_prob)
    intercept_est = np.mean(intercept_samples_pct)

    # mean and hdi for intercept + slope
    mu_slope_samples = trace.posterior["mu_slope"].values.flatten()
    intercept_slope_samples_pct = logistic_to_percent(
        mu_intercept_samples + mu_slope_samples
    )
    intercept_slope_hdi = az.hdi(intercept_slope_samples_pct, hdi_prob=hdi_prob)
    intercept_slope_est = np.mean(intercept_slope_samples_pct)

    # mean and hdi for slope
    slope_effect = intercept_slope_samples_pct - intercept_samples_pct
    slope_effect_hdi = az.hdi(slope_effect, hdi_prob=hdi_prob)
    slope_effect_est = np.mean(slope_effect)

    # mean and hdi for intercepts
    # collapse the first dimension (number of chains) and we have 10 regions
    random_intercepts_samples = trace.posterior["intercepts"].values.reshape(
        -1, n_regions
    )
    random_intercepts_pct = logistic_to_percent(random_intercepts_samples)
    random_intercepts_hdi = az.hdi(random_intercepts_pct, hdi_prob=hdi_prob)
    random_intercepts_est = np.mean(random_intercepts_pct, axis=0)

    # mean and hdi for intercept + slopes
    random_slopes_samples = trace.posterior["slopes"].values.reshape(-1, n_regions)
    random_intercepts_slopes_pct = logistic_to_percent(
        random_intercepts_samples + random_slopes_samples
    )
    random_intercepts_slopes_hdi = az.hdi(
        random_intercepts_slopes_pct, hdi_prob=hdi_prob
    )
    random_intercepts_slopes_est = np.mean(random_intercepts_slopes_pct, axis=0)

    # mean and hdi for slopes
    random_slope_effect = random_intercepts_slopes_pct - random_intercepts_pct
    random_slope_effect_hdi = az.hdi(random_slope_effect, hdi_prob=hdi_prob)
    random_slope_effect_est = np.mean(random_slope_effect, axis=0)

    # construct summary of this;
    summary_main_effects = pd.DataFrame(
        columns=["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"],
        data=[
            ("Intercept", intercept_est, intercept_hdi[0], intercept_hdi[1]),
            (
                "Intercept + Slope",
                intercept_slope_est,
                intercept_slope_hdi[0],
                intercept_slope_hdi[1],
            ),
            ("Slope", slope_effect_est, slope_effect_hdi[0], slope_effect_hdi[1]),
        ],
    )

    summary_random_intercepts = pd.DataFrame(
        columns=["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"],
        data=[
            (
                f"intercept_{i}",
                random_intercepts_est[i],
                random_intercepts_hdi[i][0],
                random_intercepts_hdi[i][1],
            )
            for i in range(n_regions)
        ],
    )

    summary_random_slopes = pd.DataFrame(
        columns=["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"],
        data=[
            (
                f"slope_{i}",
                random_slope_effect_est[i],
                random_slope_effect_hdi[i][0],
                random_slope_effect_hdi[i][1],
            )
            for i in range(n_regions)
        ],
    )

    summary_random_intercepts_slopes = pd.DataFrame(
        columns=["parameter", "mean_pct", "hdi_2.5%_pct", "hdi_97.5%_pct"],
        data=[
            (
                f"intercept + slope_{i}",
                random_intercepts_slopes_est[i],
                random_intercepts_slopes_hdi[i][0],
                random_intercepts_slopes_hdi[i][1],
            )
            for i in range(n_regions)
        ],
    )

    summary_pct = pd.concat(
        [
            summary_main_effects,
            summary_random_intercepts,
            summary_random_slopes,
            summary_random_intercepts_slopes,
        ]
    )

    summary_pct.to_csv(f"../mdl_output/pct/{label}_summary_pct.csv", index=False)
