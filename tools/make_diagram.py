"""Renders figures/architecture.png — architecture diagram / writeup cover."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "architecture.png"

INK = "#1a2733"
BLUE = "#dbeafe"
GREEN = "#dcfce7"
ORANGE = "#ffedd5"
RED = "#fee2e2"
GRAY = "#f1f5f9"


def box(ax, x, y, w, h, label, fc, fontsize=10, weight="bold", sub=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fc, ec=INK, lw=1.4,
        )
    )
    cy = y + h - 0.32 if sub else y + h / 2
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=INK)
    if sub:
        ax.text(x + w / 2, y + (h - 0.5) / 2, sub, ha="center", va="center",
                fontsize=8.2, color=INK)


def arrow(ax, xy_from, xy_to, label="", color=INK, ls="-", lw=1.6, label_dy=0.14):
    ax.add_patch(
        FancyArrowPatch(
            xy_from, xy_to, arrowstyle="-|>", mutation_scale=14,
            color=color, lw=lw, linestyle=ls, shrinkA=4, shrinkB=4,
        )
    )
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2
        my = (xy_from[1] + xy_to[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.2,
                color=color, fontstyle="italic")


fig, ax = plt.subplots(figsize=(12.5, 7.2))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 7.2)
ax.axis("off")

ax.text(6.25, 6.9, "Overshoot Governor", ha="center", fontsize=17,
        fontweight="bold", color=INK)
ax.text(6.25, 6.5, "Token-budget admission control for multi-agent systems — "
                   "designed with Donella Meadows' leverage points",
        ha="center", fontsize=10, color=INK)

# --- ADK Runner with the agent team and the spawn/lease tree
box(ax, 0.4, 1.1, 4.6, 4.7, "", GRAY)
ax.text(2.7, 5.45, "ADK 2.0 Runner", ha="center", fontsize=11,
        fontweight="bold", color=INK)

box(ax, 1.5, 4.3, 2.4, 0.75, "Coordinator", BLUE, sub="LlmAgent")
box(ax, 0.7, 3.0, 1.7, 0.7, "Researcher", BLUE, fontsize=9)
box(ax, 2.9, 3.0, 1.7, 0.7, "Writer", BLUE, fontsize=9)
arrow(ax, (2.3, 4.3), (1.6, 3.7))
arrow(ax, (3.1, 4.3), (3.7, 3.7))

# lease tree
box(ax, 0.7, 1.35, 3.9, 1.15, "Quota lease tree  (#7 loop gain)", GREEN,
    fontsize=9,
    sub="spawn_child() carves from parent's remainder\n"
        "Σ children ≤ parent → cascades self-extinguish")
arrow(ax, (1.55, 3.0), (1.7, 2.5), lw=1.2)
arrow(ax, (3.75, 3.0), (3.6, 2.5), lw=1.2)

# --- Governor plugin
box(ax, 5.6, 2.6, 3.3, 3.2, "", ORANGE)
ax.text(7.25, 5.45, "BudgetGovernorPlugin", ha="center", fontsize=11,
        fontweight="bold", color=INK)
ax.text(7.25, 5.12, "registered once on the Runner —\ncovers every agent & subagent",
        ha="center", fontsize=7.8, color=INK, fontstyle="italic")
box(ax, 5.85, 3.95, 2.8, 0.85, "AtomicLedger  (#8, #9)", "white", fontsize=9,
    sub="reserve → execute → reconcile\nspent + committed ≤ budget")
box(ax, 5.85, 3.15, 2.8, 0.62, "OutputEstimator — p90 per agent", "white", fontsize=8.4)
box(ax, 5.85, 2.75, 2.8, 0.32, "completion reserve (#11)", "white", fontsize=8.2)

# --- Gemini
box(ax, 10.0, 4.4, 2.1, 0.95, "Gemini API", RED, sub="the finite resource")

# --- MCP server
box(ax, 5.6, 0.6, 3.3, 1.3, "MCP server", GREEN, fontsize=10,
    sub="same ledger over the standard protocol:\n"
        "reserve / settle / budget_status\n"
        "→ cross-runtime governance")
arrow(ax, (7.25, 1.9), (7.25, 2.6), lw=1.2)

# --- other runtimes via MCP
box(ax, 10.0, 0.7, 2.1, 1.1, "Other runtimes", GRAY, fontsize=9,
    sub="Claude Code,\nscripts, CI agents")
arrow(ax, (10.0, 1.25), (8.9, 1.25), lw=1.2)

# --- main flows
arrow(ax, (5.0, 4.6), (5.6, 4.5), "before_model:\nreserve (atomic)", label_dy=0.22)
arrow(ax, (8.9, 4.6), (10.0, 4.8), "admitted calls\nonly")
arrow(ax, (10.0, 4.55), (8.9, 4.15), "usage_metadata", label_dy=-0.34)
arrow(ax, (5.6, 3.3), (5.0, 3.6), "the meter (#6):\nbudget state into context",
      color="#166534", ls="--", label_dy=-0.52)

ax.text(6.25, 0.15,
        "Meadows leverage points:  #11 buffers · #9 delays · #8 balancing loops · "
        "#7 reinforcing-loop gain · #6 information flows",
        ha="center", fontsize=8.5, color=INK)

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
