"""
Flowchart comparing v5 and v6 DRH → ASJP language mapping pipelines.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── colours ──────────────────────────────────────────────────────────────────
C_SHARED  = "#4C72B0"   # blue  – steps common to both pipelines
C_V5      = "#DD8452"   # orange – v5-specific
C_V6      = "#55A868"   # green  – v6-specific
C_LOSS    = "#C44E52"   # red    – loss boxes
C_TEXT    = "white"
C_LOSS_TXT= "white"

# ── layout constants ──────────────────────────────────────────────────────────
BOX_W, BOX_H = 0.32, 0.07   # box width / height (axes-fraction)
LOSS_W        = 0.20
XS, XV5, XV6  = 0.50, 0.25, 0.75   # x-centres: shared, v5, v6
Y_START       = 0.95
Y_STEP        = 0.17           # vertical distance between main boxes
LOSS_DX       = 0.22           # horizontal offset for loss boxes


def add_box(ax, x, y, text, color, width=BOX_W, height=BOX_H, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor=color, transform=ax.transAxes, clip_on=False, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=C_TEXT, transform=ax.transAxes, zorder=4,
            fontweight='bold', wrap=False)


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.5):
    ax.annotate("",
        xy=(x2, y2), xycoords='axes fraction',
        xytext=(x1, y1), textcoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
        zorder=2)


def loss_box(ax, x_main, y_main, side, label, color=C_LOSS):
    """Draw a loss box to the side of (x_main, y_main)."""
    dx = LOSS_DX if side == 'right' else -LOSS_DX
    xl = x_main + dx
    yl = y_main
    # dashed connector from main box edge to loss box
    edge_x = x_main + (BOX_W / 2 if side == 'right' else -BOX_W / 2)
    lx0 = xl - LOSS_W / 2 if side == 'right' else xl + LOSS_W / 2
    ax.annotate("",
        xy=(lx0, yl), xycoords='axes fraction',
        xytext=(edge_x, yl), textcoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                        linestyle='dashed'),
        zorder=2)
    add_box(ax, xl, yl, label, color, width=LOSS_W, height=0.06, fontsize=8)


# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 10))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')

fig.suptitle("DRH → ASJP language mapping: v5 vs v6", fontsize=13,
             fontweight='bold', y=0.98)

# column headers
ax.text(XV5, 0.97, "v5  (direct name match)", ha='center', va='center',
        fontsize=10, color=C_V5, fontweight='bold',
        transform=ax.transAxes)
ax.text(XV6, 0.97, "v6  (via Glottolog languoid.csv)", ha='center', va='center',
        fontsize=10, color=C_V6, fontweight='bold',
        transform=ax.transAxes)

# ── shared step 1 ─────────────────────────────────────────────────────────────
y1 = Y_START - 0.06
add_box(ax, XS, y1, "DRH entries with answers\nn = 828", C_SHARED)

# ── shared step 2 ─────────────────────────────────────────────────────────────
y2 = y1 - Y_STEP
add_box(ax, XS, y2, "Entries with language tags\nn = 747", C_SHARED)
arrow(ax, XS, y1 - BOX_H/2, XS, y2 + BOX_H/2)

# loss: no language tag (shared)
loss_box(ax, XS, (y1 + y2) / 2, 'right',
         "No language tag\n−81 entries")

# ── split into v5 / v6 ───────────────────────────────────────────────────────
y_split = y2 - BOX_H / 2 - 0.02
# branching arrows from shared box
arrow(ax, XS, y2 - BOX_H/2, XV5, y_split + 0.02)
arrow(ax, XS, y2 - BOX_H/2, XV6, y_split + 0.02)

# ── v5 step 3: direct ASJP name match ────────────────────────────────────────
y3v5 = y2 - Y_STEP
add_box(ax, XV5, y3v5,
        "Match: DRH name\n→ ASJP Glottolog_Name\nn = 565",
        C_V5)
arrow(ax, XV5, y_split + 0.015, XV5, y3v5 + BOX_H/2)
loss_box(ax, XV5, (y_split + y3v5) / 2 + 0.01, 'left',
         "Name not in\nASJP col.\n−182 entries")

# ── v6 step 3: Glottolog intermediate ────────────────────────────────────────
y3v6 = y2 - Y_STEP
add_box(ax, XV6, y3v6,
        "Match: DRH name\n→ Glottolog → Glottocode\n→ ASJP ID\nn = 694",
        C_V6)
arrow(ax, XV6, y_split + 0.015, XV6, y3v6 + BOX_H/2)
loss_box(ax, XV6, (y_split + y3v6) / 2 + 0.01, 'right',
         "Name/code not\nin Glottolog\nor ASJP\n−53 entries")

# ── v5 step 4: map to tree tip ────────────────────────────────────────────────
y4v5 = y3v5 - Y_STEP
add_box(ax, XV5, y4v5,
        "Map ASJP ID → tree tip\n(world.tre)\nn = 555",
        C_V5)
arrow(ax, XV5, y3v5 - BOX_H/2, XV5, y4v5 + BOX_H/2)
loss_box(ax, XV5, (y3v5 + y4v5) / 2, 'left',
         "ID not in\nworld.tre\n−10 entries")

# ── v6 step 4: map to tree tip ────────────────────────────────────────────────
y4v6 = y3v6 - Y_STEP
add_box(ax, XV6, y4v6,
        "Map ASJP ID → tree tip\n(world.tre)\nn = 688",
        C_V6)
arrow(ax, XV6, y3v6 - BOX_H/2, XV6, y4v6 + BOX_H/2)
loss_box(ax, XV6, (y3v6 + y4v6) / 2, 'right',
         "ID not in\nworld.tre\n−6 entries")

# ── v5 step 5: unique tips ────────────────────────────────────────────────────
y5v5 = y4v5 - Y_STEP
add_box(ax, XV5, y5v5,
        "Unique tree tips\n(one per entry_id,\ndeepest level)\nn = 203",
        C_V5)
arrow(ax, XV5, y4v5 - BOX_H/2, XV5, y5v5 + BOX_H/2)

# ── v6 step 5: unique tips ────────────────────────────────────────────────────
y5v6 = y4v6 - Y_STEP
add_box(ax, XV6, y5v6,
        "Unique tree tips\n(one per entry_id,\ndeepest level)\nn = 268",
        C_V6)
arrow(ax, XV6, y4v6 - BOX_H/2, XV6, y5v6 + BOX_H/2)

# ── annotation: why v6 recovers more ─────────────────────────────────────────
note_y = 0.08
ax.text(0.50, note_y,
        "v6 recovers +129 entries at the ASJP merge step.\n"
        "Reason: ASJP's internal Glottolog_Name column is incomplete;\n"
        "routing through the full Glottolog languoid.csv (name → Glottocode)\n"
        "then matching to ASJP by Glottocode bypasses this gap.",
        ha='center', va='center', fontsize=8.5,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                  edgecolor='#aaaaaa', linewidth=1))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("data/flowchart_mapping.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved to preprocessing/data/flowchart_mapping.png")
