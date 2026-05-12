"""
vmp 2026-05-13
Reads hypothesis_tests.csv for each analysis, filters to fit_phylo_year,
and writes one LaTeX table per analysis to tables/.
"""

import os
import pandas as pd

ANALYSES = {
    "external":         "Violent external conflict (full sample)",
    "external_noeHRAF": "Violent external conflict (eHRAF excluded)",
    "internal":         "Violent internal conflict (full sample)",
    "internal_noeHRAF": "Violent internal conflict (eHRAF excluded)",
}

MARKER_LABELS = {
    "circumcision":               "Circumcision",
    "dress":                      "Dress",
    "extra_ritual_group_markers": "Extra-Ritual In-Group Markers",
    "food_taboos":                "Food Taboos",
    "hair":                       "Hair",
    "ornaments":                  "Ornaments",
    "permanent_scarring":         "Permanent Scarring",
    "tattoos_scarification":      "Tattoos or Scarification",
}

os.makedirs("../tables", exist_ok=True)


def fmt_evid_ratio(x):
    try:
        v = float(x)
        if v >= 8000:
            return "$>$8000"
        return f"{v:.0f}"
    except (ValueError, TypeError):
        return "$\\infty$"


def fmt_prob(x):
    v = float(x)
    if v >= 0.995:
        return "$>$0.99"
    return f"{v:.2f}"


def make_table(analysis, caption):
    df = pd.read_csv(f"../data/model/{analysis}/results/hypothesis_tests.csv")
    df = df[df["model"] == "fit_phylo_year"].copy()
    df["marker_label"] = df["marker"].map(MARKER_LABELS)
    df["evid_ratio_num"] = pd.to_numeric(df["evid_ratio"], errors="coerce").fillna(float("inf"))
    df = df.sort_values("evid_ratio_num", ascending=False).drop(columns="evid_ratio_num")

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append(
        "Marker & Estimate & 95\\% HDI & $P(\\beta > 0)$ & Evidence ratio \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        label   = row["marker_label"]
        est     = f"{row['estimate']:.2f}"
        hdi     = f"[{row['lo95']:.2f},\\ {row['hi95']:.2f}]"
        prob    = fmt_prob(row["post_prob"])
        evid    = fmt_evid_ratio(row["evid_ratio"])
        lines.append(f"{label} & {est} & {hdi} & {prob} & {evid} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\\\[4pt]\\small Estimates are log-odds coefficients from the "
        "phylogenetic + year + region model (\\texttt{fit\\_phylo\\_year}). "
        "HDI = highest density interval. Evidence ratio = posterior odds "
        "in favour of $\\beta > 0$."
    )
    lines.append("\\end{table}")

    return "\n".join(lines)


for analysis, caption in ANALYSES.items():
    tex = make_table(analysis, caption)
    out_path = f"../tables/hypothesis_{analysis}.tex"
    with open(out_path, "w") as f:
        f.write(tex)

print(f"Written {len(ANALYSES)} tables to tables/")
