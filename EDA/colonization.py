'''
Colonization proxy (police_force_other) × conflict × marker.

One figure per marker, saved separately.
Rows in each figure:  raw counts (top) | row-normalized / P(marker | conflict) (bottom)
Columns in each figure: no external police | external police present

2×2 cell axes: rows = violent_external (predictor), cols = marker (outcome)
Row normalisation gives P(marker | conflict condition) — the natural reading direction.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/preprocessed/answerset_large.csv")

markers = ['extra_ritual_group_markers', 'permanent_scarring']
row_labels = ['No conflict\n(violent_ext = 0)', 'Conflict\n(violent_ext = 1)']
col_labels  = ['No marker', 'Marker present']


def draw_matrix(ax, values, fmt_fn, cmap, vmin, vmax):
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(2):
        for j in range(2):
            txt   = fmt_fn(values[i, j])
            shade = (values[i, j] - vmin) / (vmax - vmin)
            color = 'white' if shade > 0.60 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=14, color=color, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)


for marker in markers:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    for col_idx, split_val in enumerate([0, 1]):
        sub = df[['violent_external', marker, 'police_force_other']].dropna()
        sub = sub[sub['police_force_other'] == split_val]

        ct = pd.crosstab(sub['violent_external'], sub[marker])
        for v in [0, 1]:
            if v not in ct.index:   ct.loc[v] = 0
            if v not in ct.columns: ct[v]      = 0
        ct = ct.sort_index().sort_index(axis=1).values.astype(float)

        total    = ct.sum()
        row_norm = ct / ct.sum(axis=1, keepdims=True)

        # ── top row: raw counts + % of panel total ──────────────────────────
        ax = axes[0, col_idx]
        draw_matrix(
            ax,
            ct,
            fmt_fn=lambda n: f"{int(n)}\n({n / total * 100:.1f}%)",
            cmap='Blues',
            vmin=0,
            vmax=ct.max(),
        )
        ax.set_title(f"police_force_other = {split_val}  (N = {int(total)})", fontsize=11)

        # ── bottom row: row-normalised, P(marker | conflict) ────────────────
        ax = axes[1, col_idx]
        draw_matrix(
            ax,
            row_norm,
            fmt_fn=lambda p: f"{p * 100:.1f}%",
            cmap='Blues',
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"police_force_other = {split_val}  (row-normalised)", fontsize=11)

    plt.tight_layout()
    out = f"../figures/EDA_correlation/colonization_{marker}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"saved {out}")
    plt.show()
