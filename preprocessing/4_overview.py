"""
vmp 2026-04-06
Overview of entries and tips across all four analyses.
"""

import pandas as pd
from pathlib import Path

analyses = [
    "external",
    "external_noeHRAF",
    "internal",
    "internal_noeHRAF",
]

markers = [
    "circumcision", "tattoos_scarification", "permanent_scarring",
    "extra_ritual_group_markers", "food_taboos", "hair", "dress", "ornaments",
]

base = Path("../data/model")

# ── 1. Total unique entries per analysis and overall ──────────────────────────

all_entries = set()
for analysis in analyses:
    analysis_entries = set()
    for marker in markers:
        path = base / analysis / "input" / f"{marker}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["entry_id"])
            analysis_entries.update(df["entry_id"].tolist())
    all_entries.update(analysis_entries)
    print(f"  {analysis}: {len(analysis_entries)} unique entries")

print(f"Total unique entries across all analyses: {len(all_entries)}")
