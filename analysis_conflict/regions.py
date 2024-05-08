import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from model_functions import logistic_to_percent


def subset_question(answerset, question_id):
    subset = answerset[answerset["question_id"] == question_id]
    subset = subset[["entry_id", "entry_name", "answer_value", "world_region_simple"]]
    subset = subset.drop_duplicates().dropna()
    subset["answer_value"] = subset["answer_value"].astype(int)
    subset["world_region_simple"] = pd.Categorical(subset["world_region_simple"])
    return subset


def fit_logistic_model(data):

    with pm.Model() as model:
        # Setting up a separate intercept for each region
        intercepts = pm.Normal(
            "intercepts",
            mu=0,
            sigma=5,
            shape=len(data["world_region_simple"].cat.categories),
        )

        # Get the index for each region from the categorical variable
        region_idx = pm.ConstantData(
            "region_idx", data["world_region_simple"].cat.codes
        )

        # Probability of circumcision for each observation based on its region
        p = pm.Deterministic("p", pm.math.sigmoid(intercepts[region_idx]))

        # Likelihood
        observed = pm.Binomial("y", n=1, p=p, observed=data["answer_value"])

        # Sample from the posterior
        trace = pm.sample(2000, return_inferencedata=True)

    return trace


def get_summary(trace, region_labels):
    summary = az.summary(trace, var_names=["intercepts"], hdi_prob=0.95)
    summary["label"] = region_labels
    summary = summary[["label", "mean", "sd", "hdi_2.5%", "hdi_97.5%"]]
    summary["mean_pct"] = logistic_to_percent(summary["mean"])
    summary["hdi_2.5%_pct"] = logistic_to_percent(summary["hdi_2.5%"])
    summary["hdi_97.5%_pct"] = logistic_to_percent(summary["hdi_97.5%"])
    summary["error_neg"] = summary["mean_pct"] - summary["hdi_2.5%_pct"]
    summary["error_pos"] = summary["hdi_97.5%_pct"] - summary["mean_pct"]
    return summary


def plot_summary(summary, outcome="Circumcision", outpath=False):
    summary = summary.sort_values("mean_pct")
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.errorbar(
        summary["mean_pct"],
        summary["label"],
        xerr=[summary["error_neg"], summary["error_pos"]],
        fmt="o",
        color="black",
        capsize=3,
    )
    plt.xlabel("Mean Estimate (95% HDI)")
    plt.title(f"{outcome} by Region")
    plt.grid(True)
    if outpath:
        plt.savefig(f"{outpath}/region_{outcome}.pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()


# load
entry_regions = pd.read_csv("../data/preprocessed/entry_regions.csv")
answerset = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# select and merge
entry_regions = entry_regions[["entry_id", "entry_name", "world_region"]]
answerset = answerset.merge(
    entry_regions, on="entry_id", how="inner"
)  # some we do not have

# check overall n in each region
entry_regions = answerset[["entry_id", "entry_name", "world_region"]].drop_duplicates()
region_counts = entry_regions.groupby("world_region").size()

"""
Africa: 149
Central Eurasia: 22
East Asia: 71
Europe: 139
North America: 86
Oceania-Australia: 25
South America: 43
South Asia: 76
Southeast Asia: 37
Southwest Asia: 77
"""

# make regions simpler
region_replacements = {
    "North America": "Americas",
    "South America": "Americas",
    "Central Eurasia": "Europe",
    "South Asia": "Asia",
    "East Asia": "Asia",
    "Southeast Asia": "Asia",
    "Southwest Asia": "Asia",
}

answerset["world_region_simple"] = answerset["world_region"].replace(
    region_replacements
)

# check amount of data now
entry_regions = answerset[
    ["entry_id", "entry_name", "world_region_simple"]
].drop_duplicates()
entry_regions.groupby("world_region_simple").size()

"""
Africa: 149
Americas: 129
Asia: 261
Europe: 161
Oceania-Australia: 25
"""

# circumcision
circumcision = subset_question(answerset, 5163)
circumcision_trace = fit_logistic_model(circumcision)
circumcision_summary = get_summary(
    circumcision_trace, circumcision["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    circumcision_summary, outcome="Circumcision", outpath="../figures/supplementary"
)
circumcision_summary.to_csv("../tables/region_circumcision.csv", index=False)

# cultural contact
cultural_contact = subset_question(answerset, 4654)
cultural_trace = fit_logistic_model(cultural_contact)
cultural_summary = get_summary(
    cultural_trace, cultural_contact["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    cultural_summary, outcome="Cultural Contact", outpath="../figures/supplementary"
)
cultural_contact.to_csv("../tables/region_cultural_contact.csv", index=False)

# permanent scarring
scarring = subset_question(answerset, 5130)
scarring_trace = fit_logistic_model(scarring)
scarring_summary = get_summary(
    scarring_trace, scarring["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    scarring_summary, outcome="Permanent Scarring", outpath="../figures/supplementary"
)
scarring_summary.to_csv("../tables/region_scarring.csv", index=False)

# tattoos/scarrification
tattoos = subset_question(answerset, 5162)
tattoos_trace = fit_logistic_model(tattoos)
tattoos_summary = get_summary(
    tattoos_trace, tattoos["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    tattoos_summary,
    outcome="Tattoos Scarification",
    outpath="../figures/supplementary",
)
tattoos_summary.to_csv("../tables/region_tattoos.csv", index=False)

# internal violent conflict
internal_violence = subset_question(answerset, 4658)
internal_trace = fit_logistic_model(internal_violence)
internal_summary = get_summary(
    internal_trace, internal_violence["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    internal_summary, outcome="Internal Violence", outpath="../figures/supplementary"
)
internal_summary.to_csv("../tables/region_internal_violence.csv", index=False)

# external violence
external_violence = subset_question(answerset, 4659)
external_trace = fit_logistic_model(external_violence)
external_summary = get_summary(
    external_trace, external_violence["world_region_simple"].cat.categories.tolist()
)
plot_summary(
    external_summary, outcome="External Violence", outpath="../figures/supplementary"
)
external_summary.to_csv("../tables/region_external_violence.csv", index=False)
