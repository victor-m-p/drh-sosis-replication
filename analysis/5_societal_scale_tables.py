"""
Format 5_societal_scale.Rmd's saved CSVs (data/model/societal_scale/results) into LaTeX
tables (data/model/societal_scale/tables). diagnostics.csv is excluded — convergence QA,
not a paper table. state_hypothesis.csv is only read for the PPs in the main-text table.
"""

import pandas as pd
from pathlib import Path
from helper_functions import write_latex_table

IN = Path("../data/model/societal_scale/results")
OUT = Path("../data/model/societal_scale/tables")
OUT.mkdir(parents=True, exist_ok=True)

# "state" reproduces the \label the manuscript already cites as tab:state_fixed_effects
PIPELINE = "state"

# model names are "outcome ~ predictors"; both tables show only the outcome, written once per
# block with a \midrule between blocks. The dash in the conflict column tells the two models of
# a block apart
fe = pd.read_csv(IN / "state_fixed_effects.csv")
fe["Outcome"] = fe["Model"].str.split(" ~ ").str[0]
block_end = [i for i in range(len(fe) - 1) if fe["Outcome"][i] != fe["Outcome"][i + 1]]
fe["Outcome"] = fe["Outcome"].where(fe["Outcome"].ne(fe["Outcome"].shift()), "")
fe = fe.fillna("---")

# main text: coefficients plus the PP for the state hypothesis
hyp = pd.read_csv(IN / "state_hypothesis.csv")
pp = hyp[hyp["hypothesis"] == "state < 0"][["Model", "post_prob"]]
main = fe.merge(pp, on="Model", how="left")
assert main["post_prob"].notna().all()
main = main[["Outcome", "N", "state", "post_prob", "violent_external"]]
write_latex_table(
    main, OUT / "main.tex",
    caption=(
        "Statehood models: coefficients (log odds) with 95\\% credible intervals, and the "
        "posterior probability that the state coefficient is negative (PP). All models include "
        "start year and a world-region random intercept. The two models for each marker are "
        "fitted on the same entries. The external conflict coefficient is credible in both "
        "models that include it (PP $>$ .99). A dash marks a term not included in the model."
    ),
    label=f"tab:{PIPELINE}_main",
    col_labels={"state": "State", "post_prob": "PP", "violent_external": "Conflict"},
    midrule_after=block_end,
)

# SI: full fixed effects
si = fe[["Outcome", "N", "Intercept", "state", "violent_external", "year_scaled"]]
write_latex_table(
    si, OUT / "fixed_effects.tex",
    caption=(
        "Fixed effects for the statehood models (95\\% credible intervals). A dash marks a "
        "term not included in the model. In the first model, external violent conflict is "
        "the outcome rather than a predictor."
    ),
    label=f"tab:{PIPELINE}_fixed_effects",
    col_labels={"state": "State", "violent_external": "External conflict",
                "year_scaled": "Start year"},
    midrule_after=block_end,
    font_size="\\footnotesize",
)
