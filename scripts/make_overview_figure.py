"""Fig. 1: the workflow of a KB-grounded AI tutor with a hallucination guard, drawn as an
open, box-free flow of icons, and where this study's labelled test statements enter.
  python3 scripts/make_overview_figure.py figures
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle,
                                Polygon, Wedge)

plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"], "font.size": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED = "#2b2b2b", "#6f6f6b"
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2a78d6", "#eb6834", "#1a9e6e", "#c43d3c", "#6f6f6b", "#6a5acd"
out = sys.argv[1] if len(sys.argv) > 1 else "."

fig, ax = plt.subplots(figsize=(190 / 25.4, 190 / 25.4 * 0.54))   # 190 mm = full page width
ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.set_aspect("equal"); ax.axis("off")


def arrow(x1, y1, x2, y2, color=INK, ls="-", lw=1.2, rad=0.0, ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))


def text(x, y, s, size=6.4, color=INK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, style=style,
            linespacing=1.25, zorder=10)


def bubble(x, y, w, h, s, tail="left"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.6",
                                fc="white", ec=MUTED, lw=0.8, zorder=5))
    if tail == "left":
        ax.add_patch(Polygon([(x + 0.3, y + 1.6), (x - 1.4, y + 0.4), (x + 0.3, y + 0.6)],
                             closed=True, fc="white", ec=MUTED, lw=0.8, zorder=5))
        ax.plot([x + 0.32, x + 0.32], [y + 0.65, y + 1.55], color="white", lw=1.5, zorder=6)
    text(x + w / 2, y + h / 2, s, size=7, style="italic")


# ------------------------------------------------------------------ icons
def avatar(cx, cy, s=1.0):
    ax.add_patch(Wedge((cx, cy - 2.6 * s), 2.6 * s, 0, 180, fc="#7a6fd0", ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy + 0.6 * s), 1.75 * s, fc="#3a2a22", ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy + 0.2 * s), 1.45 * s, fc="#f3c9a6", ec="none", zorder=6))
    ax.add_patch(Wedge((cx, cy + 0.55 * s), 1.6 * s, 20, 160, fc="#3a2a22", ec="none", zorder=7))
    for dx in (-0.5, 0.5):
        ax.add_patch(Circle((cx + dx * s, cy + 0.15 * s), 0.17 * s, fc=INK, ec="none", zorder=8))
    ax.add_patch(Wedge((cx, cy - 0.35 * s), 0.55 * s, 200, 340, width=0.12 * s, fc=INK, ec="none", zorder=8))


def chip(cx, cy, color, s=1.0):
    ax.add_patch(FancyBboxPatch((cx - 1.9 * s, cy - 1.9 * s), 3.8 * s, 3.8 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.6 * s}", fc=color, ec="none", zorder=6))
    for k in (-1.1, 0, 1.1):
        k *= s
        for (a, b, c, d) in ((cx + k, cy + 1.9 * s, cx + k, cy + 2.7 * s), (cx + k, cy - 1.9 * s, cx + k, cy - 2.7 * s),
                             (cx + 1.9 * s, cy + k, cx + 2.7 * s, cy + k), (cx - 1.9 * s, cy + k, cx - 2.7 * s, cy + k)):
            ax.plot([a, c], [b, d], color=color, lw=1.0, zorder=6)
    text(cx, cy, "AI", size=8.0, color="white", weight="bold")


def guard_icon(cx, cy):
    pts = [(cx - 2.6, cy + 2.9), (cx + 2.6, cy + 2.9), (cx + 2.6, cy + 0.3), (cx, cy - 3.3), (cx - 2.6, cy + 0.3)]
    ax.add_patch(Polygon(pts, closed=True, fc=RED, ec="none", zorder=6))
    ax.add_patch(Circle((cx - 0.3, cy + 0.6), 1.15, fc="white", ec="white", lw=0.5, zorder=7))
    ax.add_patch(Circle((cx - 0.3, cy + 0.6), 0.75, fc=RED, ec="none", zorder=8))
    ax.plot([cx + 0.45, cx + 1.4], [cy - 0.15, cy - 1.1], color="white", lw=2.0, solid_capstyle="round", zorder=8)


def cylinder(x, y, w, h, color):
    ax.add_patch(Ellipse((x + w / 2, y), w, 2.2, fc="#dbe8fa", ec=color, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, fc="#dbe8fa", ec="none", zorder=3.5))
    ax.plot([x, x], [y, y + h], color=color, lw=1.0, zorder=4)
    ax.plot([x + w, x + w], [y, y + h], color=color, lw=1.0, zorder=4)
    ax.add_patch(Ellipse((x + w / 2, y + h), w, 2.2, fc="#eef4fc", ec=color, lw=1.0, zorder=5))


def slides_icon(x, y):
    for k in range(3):
        ax.add_patch(Rectangle((x + k * 0.8, y - k * 0.8), 6.0, 4.4, fc="white", ec=INK, lw=0.7, zorder=3 + k))
    bx, by = x + 1.6, y - 1.6
    ax.add_patch(Rectangle((bx + 0.5, by + 1.9), 1.9, 1.6, fc="#9cc3ef", ec="none", zorder=7))
    for j in range(3):
        ax.plot([bx + 2.8, bx + 5.3], [by + 3.2 - j * 0.65] * 2, color=MUTED, lw=0.7, zorder=7)


def badge(cx, cy, ok, r=1.0):
    ax.add_patch(Circle((cx, cy), r, fc=GREEN if ok else RED, ec="white", lw=0.8, zorder=8))
    if ok:
        ax.plot([cx - 0.5 * r, cx - 0.1 * r, cx + 0.55 * r], [cy, cy - 0.4 * r, cy + 0.45 * r],
                color="white", lw=1.2, zorder=9)
    else:
        d = 0.4 * r
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color="white", lw=1.2, zorder=9)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color="white", lw=1.2, zorder=9)


def doc_icon(x, y, ok):
    ax.add_patch(Polygon([(x, y), (x + 3.8, y), (x + 3.8, y + 4.0), (x + 2.9, y + 4.9), (x, y + 4.9)],
                         closed=True, fc="white", ec=INK, lw=0.8, zorder=6))
    for j in range(4):
        ax.plot([x + 0.6, x + 3.1], [y + 3.9 - j * 0.8] * 2, color=MUTED, lw=0.7, zorder=7)
    badge(x + 3.6, y + 0.4, ok, r=1.05)


# ------------------------------------------------------------------ the tutor's workflow (top)
avatar(6.0, 40.0, 1.35)
text(6.0, 33.4, "Student", size=7.5, weight="bold")
bubble(11.0, 44.2, 15.0, 7.4, "\u201cHow does a\nwhile loop work?\u201d")
arrow(11.2, 38.8, 24.6, 38.8)

chip(29.0, 39.4, PURPLE, 1.2)
text(29.0, 33.4, "Tutor LLM", size=7.5, weight="bold")
arrow(34.6, 38.8, 53.8, 38.8)
text(44.2, 43.6, "draft answer:\n\u201c\u2026a while loop always runs\nits body at least once\u2026\u201d",
     size=7, style="italic", color=MUTED)

guard_icon(59.0, 39.4)
text(59.0, 33.2, "Hallucination guard", size=7.5, weight="bold")

arrow(63.6, 41.0, 77.8, 46.6, color=GREEN)
text(69.6, 46.4, "consistent", size=7, color=GREEN, weight="bold")
doc_icon(79.0, 43.4, ok=True)
text(85.6, 49.6, "Answer shown", size=7.5, weight="bold", color=GREEN, ha="left")
text(85.6, 46.6, "as written", size=7, color=MUTED, ha="left")
arrow(63.6, 37.2, 77.8, 31.6, color=RED)
text(69.6, 31.0, "conflict", size=7, color=RED, weight="bold")
doc_icon(79.0, 27.0, ok=False)
text(85.6, 33.2, "Answer flagged", size=7.5, weight="bold", color=RED, ha="left")
text(85.6, 28.8, "\u201cA while loop checks\nits condition first.\u201d", size=7, style="italic", ha="left")

# ------------------------------------------------------------------ the KB (below the guard)
cylinder(56.0, 5.0, 6.0, 7.2, BLUE)
text(59.0, 8.4, "KB", size=7.5, weight="bold", color=BLUE)
arrow(59.0, 13.8, 59.0, 31.0, color=BLUE)
text(60.4, 22.4, "facts and\nnegations", size=7, color=BLUE, style="italic", ha="left")
slides_icon(79.0, 12.6)
text(82.2, 4.0, "Lecture slides\n(37 decks)", size=7, weight="bold")
arrow(78.6, 8.6, 62.6, 8.6)
text(70.2, 12.2, "facts + one negation\nper fact (built 5 ways)", size=7, color=BLUE, style="italic")

# ------------------------------------------------------------------ how this study tests the guard (left)
text(1.5, 26.0, "How we test the guard", size=7.5, weight="bold", color=ORANGE, ha="left")
text(1.5, 23.0, "labeled statements replace the tutor's claims", size=7, color=MUTED,
     style="italic", ha="left")
for yy, lab, col in ((18.6, "circular errors: the KB's own negations", ORANGE),
                     (13.6, "independent misconceptions\n(150 course + 85 ProgMiscon)", BLUE),
                     (7.4, "correct statements\n(136 + 85 corrections)", GRAY)):
    ax.add_patch(Circle((2.6, yy), 0.85, fc=col, ec="none", zorder=6))
    text(4.3, yy, lab, size=7, ha="left")
arrow(38.0, 21.0, 55.0, 35.6, color=ORANGE, rad=-0.22)
text(43.0, 30.2, "fed to\nthe guard", size=7, color=ORANGE, style="italic", ha="right")
arrow(55.6, 10.6, 37.4, 18.4, color=ORANGE, ls="--", lw=1.1, rad=0.06)
text(46.6, 17.4, "circular", size=7, color=ORANGE, weight="bold")

fig.tight_layout(pad=0.1)
fig.savefig(f"{out}/Figure_1.pdf")
fig.savefig(f"{out}/Figure_1_preview.png", dpi=220)
print("ok")
