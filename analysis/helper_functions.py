"""
Shared helpers for formatting model/external, model/internal, etc. results into LaTeX
tables. Marker order/highlighting mirrors markers_table_order / marker_level in the
R analysis scripts (1_external.Rmd etc.) — keep both in sync if either changes.
"""

import pandas as pd

MARKERS_TABLE_ORDER = [
    "Permanent Scarring", "Extra-Ritual In-Group Markers",
    "Circumcision", "Dress", "Food Taboos",
    "Hair", "Ornaments", "Tattoos/Scarification",
]

MARKER_LEVEL = {
    "Permanent Scarring": "parent",
    "Extra-Ritual In-Group Markers": "parent",
    "Circumcision": "sub", "Dress": "sub", "Food Taboos": "sub",
    "Hair": "sub", "Ornaments": "sub", "Tattoos/Scarification": "sub",
}

FAMILY_NAMES = {
    "IE": "Indo-European", "AA": "Afro-Asiatic", "ST": "Sino-Tibetan",
    "NC": "Niger-Congo", "An": "Austronesian", "Jap": "Japonic",
    "Alt": "Altaic", "Dra": "Dravidian", "AuA": "Austroasiatic",
    "Kor": "Koreanic", "OM": "Oto-Manguean", "TK": "Tai-Kadai",
    "Ain": "Ainu", "Aus": "Australian", "Chi": "Chibchan",
    "Chn": "Chonan", "HM": "Hmong-Mien", "Iro": "Iroquoian",
    "KT": "Kiowa-Tanoan", "Krd": "Kordofanian", "May": "Mayan",
    "NDe": "Na-Dene", "Pan": "Panoan", "Sah": "Saharan",
    "Sep": "Sepik", "Sio": "Siouan", "UA": "Uto-Aztecan", "Ura": "Uralic",
}


def order_by_marker(df, marker_col="Marker"):
    """Reorder rows to MARKERS_TABLE_ORDER. Stable sort, so marker blocks with multiple
    rows (e.g. one row per region) keep their original within-marker row order."""
    order_map = {m: i for i, m in enumerate(MARKERS_TABLE_ORDER)}
    return (df.assign(_order=df[marker_col].map(order_map))
              .sort_values("_order", kind="stable")
              .drop(columns="_order"))


def fmt(est, lo, hi):
    return f"{est:.2f} [{lo:.2f}, {hi:.2f}]"


def write_latex_table(df, out_path, caption, label, bold_marker_col=None, group_by_marker=False,
                       col_widths=None, col_labels=None, longtable=False):
    """
    Write a pandas DataFrame to a booktabs-style LaTeX table.

    bold_marker_col: column holding marker display names; rows for the two DRH top-level
        markers (MARKER_LEVEL == "parent") get their marker cell wrapped in \\textbf{}.
    group_by_marker: if True, insert a \\midrule after the last parent-marker row, to
        visually separate the two parent markers from the six sub-markers. Works whether
        each marker has one row (fixed/random effects) or several (region_effects).
    col_widths: dict of {column: LaTeX column spec}, e.g. {"Marker": "p{3cm}"}, for columns
        that should wrap instead of the default "l". Long marker names (e.g. "Extra-Ritual
        In-Group Markers") are usually what forces a table too wide for the page.
    col_labels: dict of {column: display header}, for renaming just the printed header
        without touching the underlying column name (which bold_marker_col etc. still use).
    longtable: use the longtable environment (spans multiple pages, repeating the header
        on each page) instead of table+tabular. Needs \\usepackage{longtable} in the
        document preamble — for tables with too many rows to fit on one page (e.g.
        region_effects, one row per marker x region).
    """
    col_widths = col_widths or {}
    col_labels = col_labels or {}
    col_format = "".join(col_widths.get(c, "l") for c in df.columns)
    header_row = " & ".join(col_labels.get(c, c) for c in df.columns) + " \\\\"
    n_cols = len(df.columns)

    midrule_after = None
    if bold_marker_col is not None and group_by_marker:
        is_parent = df[bold_marker_col].map(MARKER_LEVEL).eq("parent").to_numpy()
        if is_parent.any():
            midrule_after = is_parent.nonzero()[0].max()

    # raw numeric (float) columns get padded to 2 decimals, including trailing zeros
    # (e.g. 1.0 -> "1.00", 0.9 -> "0.90"); pre-formatted "est [lo, hi]" string columns and
    # int columns (N, counts) pass through unchanged
    is_float_col = {c: pd.api.types.is_float_dtype(df[c]) for c in df.columns}

    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = [f"{v:.2f}" if is_float_col[c] else str(v) for c, v in row.items()]
        if bold_marker_col is not None and MARKER_LEVEL.get(row[bold_marker_col]) == "parent":
            idx = df.columns.get_loc(bold_marker_col)
            cells[idx] = f"\\textbf{{{cells[idx]}}}"
        body.append(" & ".join(cells) + " \\\\")
        if midrule_after is not None and i == midrule_after:
            body.append("\\midrule")

    if longtable:
        lines = [
            "\\small",
            f"\\begin{{longtable}}{{{col_format}}}",
            f"\\caption{{{caption}}} \\label{{{label}}} \\\\",
            "\\toprule", header_row, "\\midrule",
            "\\endfirsthead",
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\small\\itshape (continued from previous page)}} \\\\",
            "\\toprule", header_row, "\\midrule",
            "\\endhead",
            "\\midrule",
            f"\\multicolumn{{{n_cols}}}{{r}}{{\\small\\itshape continued on next page}} \\\\",
            "\\endfoot",
            "\\bottomrule",
            "\\endlastfoot",
            *body,
            "\\end{longtable}",
        ]
    else:
        lines = [
            "\\begin{table}[!ht]", "\\centering", "\\small",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_format}}}",
            "\\toprule", header_row, "\\midrule",
            *body,
            "\\bottomrule", "\\end{tabular}", "\\end{table}",
        ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
