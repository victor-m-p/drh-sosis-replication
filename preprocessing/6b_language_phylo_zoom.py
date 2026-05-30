"""
vmp 2026-05-29
Zoomed phylogeny: IE (green) + AA (red) + NC (cyan) clades only,
annotated with pairwise covariances from English to illustrate drop-off.
"""
import colorsys
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo

sys.setrecursionlimit(5000)

# ── data: keep only IE, AA, NC ────────────────────────────────────────────────
tip_map = pd.read_csv("../data/preprocessed/tip_map.csv")
entry_counts = tip_map.groupby("tip_name").size().rename("n")

all_tips  = tip_map["tip_name"].dropna().unique()
our_tips  = set(t for t in all_tips if t.split(".")[0] == "IE")
tip_family = {t: t.split(".")[0] for t in our_tips}

A_corr = pd.read_csv("../data/preprocessed/phylo_corr.csv", index_col=0)

# ── prune tree ────────────────────────────────────────────────────────────────
tree = Phylo.read("../data/jaeger2018/world.tre", "newick")

def prune_to_tips(clade, keep):
    if clade.is_terminal():
        return clade if clade.name in keep else None
    survivors = [prune_to_tips(c, keep) for c in clade.clades]
    survivors = [s for s in survivors if s is not None]
    if not survivors:
        return None
    if len(survivors) == 1:
        s = survivors[0]
        s.branch_length = (s.branch_length or 0) + (clade.branch_length or 0) or None
        return s
    clade.clades = survivors
    return clade

root = prune_to_tips(tree.root, our_tips)

# ── layout ────────────────────────────────────────────────────────────────────
tip_y = {}
_y = [0]

def assign_y(clade):
    if clade.is_terminal():
        tip_y[clade.name] = _y[0]; _y[0] += 1
    else:
        for child in clade.clades: assign_y(child)

assign_y(root)

node_mid, node_dep = {}, {}

def layout(clade, depth=0):
    if clade.is_terminal():
        node_mid[id(clade)] = tip_y[clade.name]
        node_dep[id(clade)] = depth
    else:
        for child in clade.clades: layout(child, depth + 1)
        ys = [node_mid[id(c)] for c in clade.clades]
        node_mid[id(clade)] = (min(ys) + max(ys)) / 2
        node_dep[id(clade)] = depth

layout(root)
max_dep = max(node_dep.values())

# ── colors ────────────────────────────────────────────────────────────────────
FAM_COLOR = {
    "IE": colorsys.hsv_to_rgb(0.38, 0.72, 0.62),   # green
    "AA": colorsys.hsv_to_rgb(0.02, 0.72, 0.72),   # red
    "NC": colorsys.hsv_to_rgb(0.55, 0.72, 0.72),   # cyan
}
GREY = (0.72, 0.72, 0.72)

clade_fam = {}

def precomp_fam(clade):
    if clade.is_terminal():
        clade_fam[id(clade)] = tip_family[clade.name]
    else:
        for child in clade.clades: precomp_fam(child)
        child_fams = {clade_fam[id(c)] for c in clade.clades}
        clade_fam[id(clade)] = child_fams.pop() if len(child_fams) == 1 else None

precomp_fam(root)

# ── draw ──────────────────────────────────────────────────────────────────────
n_tips = len(our_tips)
max_n  = entry_counts.max()

TREE_SCALE = 0.1   # tips land at x=TREE_SCALE; shrink to compress branches

fig, ax = plt.subplots(figsize=(6, n_tips * 0.08 + 1))

def draw(clade, parent_x=None):
    dep = node_dep[id(clade)]
    cy  = node_mid[id(clade)]
    cx  = TREE_SCALE if clade.is_terminal() else dep / max_dep * TREE_SCALE

    fam = clade_fam[id(clade)]
    c   = FAM_COLOR.get(fam, GREY) if fam else GREY

    if parent_x is not None:
        ax.plot([parent_x, cx], [cy, cy], color=c, lw=0.9, alpha=0.85)

    if clade.is_terminal():
        n    = entry_counts.get(clade.name, 1)
        size = 6 + 55 * (np.sqrt(n) / np.sqrt(max_n))
        ax.scatter([cx], [cy], s=size, color=c, zorder=3, alpha=0.9, linewidths=0)
        label = clade.name.split(".")[-1] + f"  ({n})"
        ax.text(cx + 0.005, cy, label, fontsize=6, va="center", color=c)
    else:
        ys = [node_mid[id(ch)] for ch in clade.clades]
        ax.plot([cx, cx], [min(ys), max(ys)], color=c, lw=0.9, alpha=0.85)
        for child in clade.clades: draw(child, cx)

draw(root)

# ── covariance brackets ───────────────────────────────────────────────────────
ENGLISH = "IE.GERMANIC.ENGLISH"
pairs = [
    ("IE.ROMANCE.FRENCH",  "#232b2b", "Eng–French"),   # inner (large span)
    ("IE.GERMANIC.GOTHIC", "#232b2b", "Eng–Gothic"),   # outer (tiny span, label stays near English)
]

TICK  = 0.007
BX0   = TREE_SCALE + 0.06   # just past the longest tip label
BSTEP = 0.012

for k, (t2, c, label) in enumerate(pairs):
    bx   = BX0 + k * BSTEP
    y1   = tip_y[ENGLISH]
    y2   = tip_y[t2]
    corr = A_corr.loc[ENGLISH, t2]
    ymid = (y1 + y2) / 2

    ax.plot([bx, bx], [min(y1, y2), max(y1, y2)], color=c, lw=1.5, alpha=0.9, zorder=6)
    ax.plot([bx - TICK, bx], [y1, y1], color=c, lw=1.5, alpha=0.9, zorder=6)
    ax.plot([bx - TICK, bx], [y2, y2], color=c, lw=1.5, alpha=0.9, zorder=6)
    ax.text(bx + 0.006, ymid, f"{label}  A={corr:.2f}",
            fontsize=6, va="center", color=c, fontweight="bold")

# Mark English with a star
ax.scatter([TREE_SCALE], [tip_y[ENGLISH]], s=60, marker="*",
           color="black", zorder=7, linewidths=0)

#ax.set_xlim(-0.01, BX0 + len(pairs) * BSTEP + 0.12)  # auto-tight right margin
ax.axis("off")

plt.tight_layout()
plt.savefig("../figures/phylo/language_phylo_zoom.png", dpi=300, bbox_inches="tight")
plt.close()
