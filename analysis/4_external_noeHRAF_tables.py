"""
Format 4_external_noeHRAF.Rmd's saved CSVs (data/model/external_noeHRAF/results) into
LaTeX tables (data/model/external_noeHRAF/tables). diagnostics_* are excluded —
convergence QA, not a paper table.
"""

import pandas as pd
from pathlib import Path
from helper_functions import order_by_marker, fmt, write_latex_table

IN = Path("../data/model/external_noeHRAF/results")
OUT = Path("../data/model/external_noeHRAF/tables")
OUT.mkdir(parents=True, exist_ok=True)

# short, unique-across-pipelines tag for \label{} keys (fixed_effects_baseline etc. would
# otherwise collide with the same table in 1_external_tables.py and friends); not shown in
# captions since the paper's section structure already makes clear which analysis is which
PIPELINE = "extnoehraf"

# fixed effects
for suffix in ["baseline", "phylo"]:
    df = pd.read_csv(IN / f"fixed_effects_{suffix}.csv")
    df = order_by_marker(df)
    write_latex_table(
        df, OUT / f"fixed_effects_{suffix}.tex",
        caption=f"Fixed effects, {suffix} model, excluding eHRAF (95\% credible intervals).",
        label=f"tab:{PIPELINE}_fixed_effects_{suffix}",
        bold_marker_col="Marker", group_by_marker=True,
        col_widths={"Marker": "p{4cm}"},
        col_labels={"violent_external": "External Conflict", "year_scaled": "Start Year"},
    )

# random effects
for suffix in ["baseline", "phylo"]:
    df = pd.read_csv(IN / f"random_effects_{suffix}.csv")
    df = order_by_marker(df)
    write_latex_table(
        df, OUT / f"random_effects_{suffix}.tex",
        caption=f"Random effects, {suffix} model, excluding eHRAF.",
        label=f"tab:{PIPELINE}_random_effects_{suffix}",
        bold_marker_col="Marker", group_by_marker=True,
        col_labels={"sd_world_region": "sd(Region)", "sd_phylo": "sd(Phylo)",
                    "sd_tip_name": "sd(Tip)"},
    )

# hypothesis test + AME (beta/CI are separate numeric columns in this CSV, unlike the
# tables above — format them into one "est [lo, hi]" column here)
for suffix in ["baseline", "phylo"]:
    df = pd.read_csv(IN / f"hypothesis_ame_{suffix}.csv")
    df["beta"] = [fmt(b, lo, hi) for b, lo, hi in zip(df["beta"], df["ci_lo"], df["ci_hi"])]
    df["AME"] = [fmt(a, lo, hi) for a, lo, hi in zip(df["AME"], df["AME_lo"], df["AME_hi"])]
    df = df[["Marker", "N", "beta", "post_prob", "AME"]]
    df = order_by_marker(df)
    write_latex_table(
        df, OUT / f"hypothesis_ame_{suffix}.tex",
        caption=f"Hypothesis test ($\\beta > 0$) and average marginal effect, {suffix} model, excluding eHRAF.",
        label=f"tab:{PIPELINE}_hypothesis_ame_{suffix}",
        bold_marker_col="Marker", group_by_marker=True,
        col_labels={"beta": "$\\beta$ (log odds)", "post_prob": "PP", "AME": "AME [95\\% CI]"},
    )

# phylogenetic signal
df = pd.read_csv(IN / "phylo_signal.csv")
df = order_by_marker(df)
write_latex_table(
    df, OUT / "phylo_signal.tex",
    caption=(
        "Phylogenetic signal by marker, excluding eHRAF. Proportion of tip-level "
        "variance that is tree-structured, computed per posterior draw as "
        "$\\sigma^2_{\\mathrm{phylo}} / (\\sigma^2_{\\mathrm{phylo}} + \\sigma^2_{\\mathrm{tip}})$ "
        "on the logit scale. This is not Pagel's $\\lambda$."
    ),
    label=f"tab:{PIPELINE}_phylo_signal",
    bold_marker_col="Marker", group_by_marker=True,
    col_labels={"phylo_signal": "$\\sigma^2_{\\mathrm{phylo}} / "
                                "(\\sigma^2_{\\mathrm{phylo}} + \\sigma^2_{\\mathrm{tip}})$"},
)
