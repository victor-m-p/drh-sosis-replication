"""
vmp 2026-05-30
Coarse-grained phylogeny: each contiguous monofamilial clade collapsed to
one node. Node size = total DRH entries in that clade; label = most common
tip within the clade (random tiebreak) + entry count.
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

# ── compute family for each clade ─────────────────────────────────────────────
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

# ── collect family nodes (virtual tips in the coarse tree) ────────────────────
# A family node is any clade where clade_fam is non-None but its parent is None.
# We stop recursing at these — the whole subtree collapses to one point.

def get_leaves(clade):
    if clade.is_terminal():
        return [clade.name]
    leaves = []
    for c in clade.clades:
        leaves.extend(get_leaves(c))
    return leaves

family_nodes = []  # (clade, fam, leaves)

def find_family_nodes(clade):
    if clade_fam[id(clade)] is not None:
        family_nodes.append((clade, clade_fam[id(clade)], get_leaves(clade)))
    else:
        for child in clade.clades:
            find_family_nodes(child)

find_family_nodes(root)

# For each family node: total entries and best label
fam_group_info = {}
for clade, fam, leaves in family_nodes:
    total = sum(entry_counts.get(t, 0) for t in leaves)
    counts_per_tip = {t: entry_counts.get(t, 0) for t in leaves}
    best_tip = max(counts_per_tip, key=lambda t: (counts_per_tip[t], t))
    fam_group_info[id(clade)] = {
        "total": total,
        "label": best_tip.split(".")[-1] + f"  ({total})",
        "fam": fam,
    }

# ── coarse layout: one row per family node, in traversal order ────────────────
group_y = {}
_y = [0]

def assign_group_y(clade):
    if clade_fam[id(clade)] is not None:
        group_y[id(clade)] = _y[0]
        _y[0] += 1
    else:
        for child in clade.clades:
            assign_group_y(child)

assign_group_y(root)

node_mid = {}
node_dep = {}

def coarse_layout(clade, depth=0):
    if clade_fam[id(clade)] is not None:
        node_mid[id(clade)] = group_y[id(clade)]
        node_dep[id(clade)] = depth
    else:
        for child in clade.clades:
            coarse_layout(child, depth + 1)
        ys = [node_mid[id(c)] for c in clade.clades]
        node_mid[id(clade)] = (min(ys) + max(ys)) / 2
        node_dep[id(clade)] = depth

coarse_layout(root)
max_dep = max(node_dep.values()) or 1

# ── colors ────────────────────────────────────────────────────────────────────
families = sorted(set(tip_family.values()))
hues = np.linspace(0, 1, len(families), endpoint=False)
fam_color = {f: colorsys.hsv_to_rgb(h, 0.72, 0.72) for f, h in zip(families, hues)}
GREY = (0.72, 0.72, 0.72)

# ── draw ──────────────────────────────────────────────────────────────────────
n_groups = _y[0]
max_total = max(info["total"] for info in fam_group_info.values()) or 1

fig, ax = plt.subplots(figsize=(10, n_groups * 0.15 + 1))

def draw(clade, parent_x=None):
    dep  = node_dep[id(clade)]
    cy   = node_mid[id(clade)]
    fam  = clade_fam[id(clade)]
    cx   = 1.0 if fam is not None else dep / max_dep
    c    = fam_color[fam] if fam else GREY

    if parent_x is not None:
        ax.plot([parent_x, cx], [cy, cy], color=c, lw=1.0, alpha=0.85)

    if fam is not None:
        info  = fam_group_info[id(clade)]
        total = info["total"]
        size  = 20 + 200 * (np.sqrt(total) / np.sqrt(max_total))
        ax.scatter([cx], [cy], s=size, color=c, zorder=3, alpha=0.9, linewidths=0)
        ax.text(cx + 0.030, cy, info["label"], fontsize=8, va="center", color=c)
    else:
        ys = [node_mid[id(ch)] for ch in clade.clades]
        ax.plot([cx, cx], [min(ys), max(ys)], color=c, lw=1.0, alpha=0.85)
        for child in clade.clades:
            draw(child, cx)

draw(root)

ax.set_ylim(-0.5, n_groups + 0.5)
ax.axis("off")

plt.tight_layout()
plt.savefig("../figures/phylo/language_phylo_group.png", dpi=300, bbox_inches="tight")
plt.close()
