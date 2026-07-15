"""
Format 4_external_noeHRAF.Rmd's saved CSVs (data/model/external_noeHRAF/results) into
LaTeX tables (data/model/external_noeHRAF/tables). diagnostics_* are excluded —
convergence QA, not a paper table.
"""

import pandas as pd
from pathlib import Path
from helper_functions import order_by_marker, fmt, write_latex_table, FAMILY_NAMES

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
        caption=f"Fixed effects, {suffix} model, excluding eHRAF (95\% credibility intervals).",
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
    df = df[["Marker", "N", "beta", "post_prob", "AME"]]
    df = order_by_marker(df)
    write_latex_table(
        df, OUT / f"hypothesis_ame_{suffix}.tex",
        caption=f"Hypothesis test ($\\beta > 0$) and average marginal effect, {suffix} model, excluding eHRAF.",
        label=f"tab:{PIPELINE}_hypothesis_ame_{suffix}",
        bold_marker_col="Marker", group_by_marker=True,
        col_labels={"beta": "$\\beta$ (log odds)", "post_prob": "PP"},
    )

# region effects (multiple rows per marker, one per world_region)
df = pd.read_csv(IN / "region_effects_baseline.csv")
df["model_estimate"] = [fmt(e, lo, hi) for e, lo, hi in
                         zip(df["model_estimate"], df["ci_lo"], df["ci_hi"])]
df = df[["Marker", "world_region", "n", "raw_rate", "model_estimate"]]
df = order_by_marker(df)
write_latex_table(
    df, OUT / "region_effects_baseline.tex",
    caption="Per-region raw and model-estimated rates, baseline model, excluding eHRAF.",
    label=f"tab:{PIPELINE}_region_effects_baseline",
    col_labels={"world_region": "Region", "raw_rate": "Raw Rate", "model_estimate": "Estimate"},
    longtable=True,
)

# phylogenetic signal
df = pd.read_csv(IN / "phylo_signal.csv")
df = order_by_marker(df)
write_latex_table(
    df, OUT / "phylo_signal.tex",
    caption="Phylogenetic signal by marker, excluding eHRAF.",
    label=f"tab:{PIPELINE}_phylo_signal",
    bold_marker_col="Marker", group_by_marker=True,
    col_labels={"phylo_signal": "Phylo signal ($\\lambda$)"},
)

# tattoos/scarification family breakdown: single-marker deep dive, not marker-keyed,
# so no ordering/bolding applies
df = pd.read_csv(IN / "tip_effects_tattoos_scarification.csv")
df["family"] = [f"{FAMILY_NAMES[c]} ({c})" if c in FAMILY_NAMES else c for c in df["family"]]
write_latex_table(
    df, OUT / "tattoos_scarification_family.tex",
    caption="Tattoos/Scarification: raw and model-estimated rate by language family, excluding eHRAF.",
    label=f"tab:{PIPELINE}_tattoos_scarification_family",
    col_labels={"family": "Family", "total_n": "Total N", "n_tips": "Tips",
                "mean_raw_rate": "Raw Rate", "mean_estimate": "Estimate"},
)
