"""
vmp 2026-05-29
Hierarchical plot of language tips in our dataset, pruned from the full
Jäger 2018 ASJP phylogeny (world.tre).

Structure above the family level comes from the actual tree topology
(grey branches = nodes spanning multiple families).
Colors = language family where subtree is homogeneous, grey otherwise.
Dot size at each tip = number of DRH entries mapped to that tip.
"""
import colorsys
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo

sys.setrecursionlimit(5000)

# ── data ──────────────────────────────────────────────────────────────────────
tip_map = pd.read_csv("../data/preprocessed/tip_map.csv")
entry_counts = tip_map.groupby("tip_name").size().rename("n")
our_tips = set(tip_map["tip_name"].dropna().unique())
tip_family = {t: t.split(".")[0] for t in our_tips}

# ── prune tree to our 200 tips ────────────────────────────────────────────────
tree = Phylo.read("../data/jaeger2018/world.tre", "newick")

def prune_to_tips(clade, keep):
    """Recursively keep only branches that lead to a tip in `keep`.
    Collapses unary nodes by adding their branch length to the surviving child."""
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

# ── cladogram layout ──────────────────────────────────────────────────────────
# y: in-order traversal assigns one row per tip
tip_y = {}
_y = [0]

def assign_y(clade):
    if clade.is_terminal():
        tip_y[clade.name] = _y[0]
        _y[0] += 1
    else:
        for child in clade.clades:
            assign_y(child)

assign_y(root)

# depth and y-midpoint for every node
node_mid = {}
node_dep = {}

def layout(clade, depth=0):
    if clade.is_terminal():
        node_mid[id(clade)] = tip_y[clade.name]
        node_dep[id(clade)] = depth
    else:
        for child in clade.clades:
            layout(child, depth + 1)
        ys = [node_mid[id(c)] for c in clade.clades]
        node_mid[id(clade)] = (min(ys) + max(ys)) / 2
        node_dep[id(clade)] = depth

layout(root)
max_dep = max(node_dep.values())

# ── colors ────────────────────────────────────────────────────────────────────
families = sorted(set(tip_family.values()))
hues = np.linspace(0, 1, len(families), endpoint=False)
fam_color = {f: colorsys.hsv_to_rgb(h, 0.72, 0.72) for f, h in zip(families, hues)}
GREY = (0.72, 0.72, 0.72)

# for each node: family name if all descendant tips share one family, else None
clade_fam = {}

def precomp_fam(clade):
    if clade.is_terminal():
        clade_fam[id(clade)] = tip_family[clade.name]
    else:
        for child in clade.clades:
            precomp_fam(child)
        child_fams = {clade_fam[id(c)] for c in clade.clades}
        clade_fam[id(clade)] = child_fams.pop() if len(child_fams) == 1 else None

precomp_fam(root)

# ── draw ──────────────────────────────────────────────────────────────────────
n_tips = len(our_tips)
max_n = entry_counts.max()

fig, ax = plt.subplots(figsize=(14, n_tips * 0.08 + 1))

def draw(clade, parent_x=None):
    dep = node_dep[id(clade)]
    cy = node_mid[id(clade)]
    # cladogram: all tips flush right, internal nodes scaled by depth
    cx = 1.0 if clade.is_terminal() else dep / max_dep

    fam = clade_fam[id(clade)]
    c = fam_color[fam] if fam else GREY

    if parent_x is not None:
        ax.plot([parent_x, cx], [cy, cy], color=c, lw=0.75, alpha=0.85)

    if clade.is_terminal():
        n = entry_counts.get(clade.name, 1)
        size = 6 + 55 * (np.sqrt(n) / np.sqrt(max_n))
        ax.scatter([cx], [cy], s=size, color=c, zorder=3, alpha=0.9, linewidths=0)
        label = clade.name.split(".")[-1] + f"  ({n})"
        ax.text(cx + 0.012, cy, label, fontsize=5, va="center", color=c)
    else:
        ys = [node_mid[id(ch)] for ch in clade.clades]
        ax.plot([cx, cx], [min(ys), max(ys)], color=c, lw=0.75, alpha=0.85)
        for child in clade.clades:
            draw(child, cx)

draw(root)

# Family labels on the right, after the tip labels
# FAM_X = 1.55
# for fam in families:
#     ys = [tip_y[t] for t in our_tips if tip_family[t] == fam]
#     ax.text(FAM_X, float(np.mean(ys)), fam, fontsize=8, ha="left", va="center",
#             color=fam_color[fam], fontweight="bold")

ax.set_ylim(-1, n_tips + 1)
ax.axis("off")
ax.set_title("")

plt.tight_layout()
plt.savefig("../figures/phylo/language_phylo.png", dpi=300, bbox_inches="tight")
plt.close()
