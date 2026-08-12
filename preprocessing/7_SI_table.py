'''
Create huge table for SI with all entries.
Some formatting done with Claude to make foreign-language (e.g., chinese, arabic) render in LaTeX.
'''

import os
import re
import pandas as pd

os.makedirs("../tables", exist_ok=True)

# Arabic-script codepoint ranges (main block, supplement, extended-A, presentation forms).
# Adjacent Arabic words separated by single spaces are grouped into one run so the whole
# phrase gets reordered together, not word-by-word.
ARABIC_CHAR = r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
ARABIC_RE = re.compile(ARABIC_CHAR + r"+(?: +" + ARABIC_CHAR + r"+)*")

LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}

def escape_latex(text):
    return "".join(LATEX_SPECIAL[ch] if ch in LATEX_SPECIAL else ch for ch in text)

def add_longtable_heading(latex_str, heading_rows):
    # interact.cls redefines \caption for its own numbered-float system, which
    # breaks longtable's inline \caption{...} \\ header line. Sidestep it entirely
    # with plain multicolumn rows instead, shown once on the first page.
    lines = latex_str.splitlines()
    lines[1:1] = heading_rows
    return "\n".join(lines)

def wrap_footnotesize(latex_str):
    return "{\\footnotesize\n" + latex_str + "\n}"

def format_cell(text):
    # Escapes LaTeX specials and right-to-left orders Arabic-script runs.
    # True bidi reordering needs the `bidi` package (loaded by polyglossia, or
    # needed to activate XeTeX's \beginR/\endR primitives at all), and that
    # package broke interact.cls's \maketitle. So instead of relying on bidi,
    # we reverse word order ourselves: within each word the character order
    # stays untouched (font shaping/joining only depends on that adjacency,
    # not on bidi), and reversing which word comes first is all bidi would
    # otherwise do for a plain space-separated Arabic phrase. Just needs
    # \arabicfont (from fontspec), no extra package.
    # xeCJK handles CJK runs automatically, no markup required there.
    text = str(text)
    parts = []
    last_end = 0
    for m in ARABIC_RE.finditer(text):
        parts.append(escape_latex(text[last_end:m.start()]))
        words = m.group().split(" ")
        reordered = " ".join(reversed(words))
        parts.append(r"{\arabicfont " + escape_latex(reordered) + "}")
        last_end = m.end()
    parts.append(escape_latex(text[last_end:]))
    return "".join(parts)

answerset_large = pd.read_csv("../data/preprocessed/answerset_large.csv")
language_master = pd.read_csv("../data/preprocessed/language_master.csv")

# take out key columns
answerset_large = answerset_large[["entry_id", "world_region", "extra_ritual_group_markers", "permanent_scarring", "violent_external", "violent_internal"]]
language_master  = language_master[["entry_id", "entry_name", "tip_assigned"]]

# merge these first
data = answerset_large.merge(language_master, on = ["entry_id"], how = 'inner')

# missing end year + expert:
entry_raw = pd.read_csv("../data/raw/entry_data.csv")
entry_raw = entry_raw[["entry_id", "expert_name", "year_from", "year_to"]]

# let's try to put it together
data = data.merge(entry_raw, on = 'entry_id', how = 'inner')
data['year_range'] = "[" + data['year_from'].astype(str) + ", " + data['year_to'].astype(str) + "]"
data = data.drop(columns = ["year_from", "year_to"])

# recode binary markers/conflict variables as Yes/No, keep missing blank
binary_cols = ["extra_ritual_group_markers", "permanent_scarring", "violent_external", "violent_internal"]
for col in binary_cols:
    data[col] = data[col].map({0.0: "No", 1.0: "Yes"}).fillna("--")

data = data.sort_values("entry_id").reset_index(drop=True)

# escape LaTeX specials + wrap Arabic script ourselves; to_latex's escape=True
# would also mangle the \textarabic{} macros we insert, so it stays off below
text_cols = ["entry_name", "world_region", "expert_name", "tip_assigned"]
for col in text_cols:
    data[col] = data[col].apply(format_cell)

# --- Table 1: main analysis variables ---
main_vars = data[["entry_id", "entry_name", "extra_ritual_group_markers", "permanent_scarring",
                   "violent_external", "violent_internal", "expert_name"]]
main_vars = main_vars.rename(columns = {
    "entry_id": "Entry ID",
    "entry_name": "Entry Name",
    "extra_ritual_group_markers": "ERM",
    "permanent_scarring": "PS",
    "violent_external": "EC",
    "violent_internal": "IC",
    "expert_name": "Expert",
})

# ragged-right inside p{} cells, without needing the array package (which
# conflicts with interact.cls) for the >{\raggedright\arraybackslash} column modifier.
# \raggedright locally redefines \\ to \@centercr; wrapping it in its own {} group
# ensures that redefinition is undone at the closing brace, before the row's own
# trailing \\ -- otherwise the row never ends and swallows the next row's cells too.
main_vars["Entry Name"] = r"{\raggedright " + main_vars["Entry Name"] + "}"
main_vars["Expert"] = r"{\raggedright " + main_vars["Expert"] + "}"

main_vars_latex = main_vars.to_latex(
    index = False,
    escape = False,
    longtable = True,
    column_format = r"rp{3.5cm}ccccp{3cm}",
    label = "tab:si_main_variables",
)
main_vars_latex = add_longtable_heading(main_vars_latex, [
    r"\multicolumn{7}{l}{\textbf{Table S1: Main analysis variables for all entries.}} \\",
    r"\multicolumn{7}{l}{\textit{ERM = Extra-Ritual Markers; PS = Permanent Scarring; EC = External Conflict; IC = Internal Conflict}} \\",
])
main_vars_latex = wrap_footnotesize(main_vars_latex)
with open("../tables/si_main_variables.tex", "w") as f:
    f.write(main_vars_latex)

# --- Table 2: metadata / controls ---
metadata_vars = data[["entry_id", "entry_name", "world_region", "year_range", "tip_assigned"]]
metadata_vars = metadata_vars.rename(columns = {
    "entry_id": "Entry ID",
    "entry_name": "Entry Name",
    "world_region": "World Region",
    "year_range": "Year Range",
    "tip_assigned": "Language Tip",
})

metadata_vars["Entry Name"] = r"{\raggedright " + metadata_vars["Entry Name"] + "}"

# language tip codes (e.g. IE.GREEK.GREEK_ANCIENT) have no spaces, so the p{}
# column has no natural break point; allow one after each dot instead
metadata_vars["Language Tip"] = metadata_vars["Language Tip"].str.replace(".", r".\allowbreak{}", regex = False)
metadata_vars["Language Tip"] = metadata_vars["Language Tip"].str.replace(r"\_", r"\_\allowbreak{}", regex = False)
metadata_vars["Language Tip"] = r"{\raggedright " + metadata_vars["Language Tip"] + "}"

metadata_vars_latex = metadata_vars.to_latex(
    index = False,
    escape = False,
    longtable = True,
    column_format = r"rp{3.5cm}llp{3.5cm}",
    label = "tab:si_metadata",
)
metadata_vars_latex = add_longtable_heading(metadata_vars_latex, [
    r"\multicolumn{5}{l}{\textbf{Table S2: Entry metadata: region, time period, and phylogenetic language tip.}} \\",
])
metadata_vars_latex = wrap_footnotesize(metadata_vars_latex)
with open("../tables/si_metadata.tex", "w") as f:
    f.write(metadata_vars_latex)
