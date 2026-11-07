"""make_overview_figure.py -- Figure 1: study design (circular vs independent evaluation).
Block diagram in the plain style of Array (Elsevier) figures. Vector PDF, embedded fonts.
USAGE: python3 scripts/make_overview_figure.py figures   (writes figures/Figure_1.pdf)"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({"font.family": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
W, H = 6.7, 3.35
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
INK, FILL, GFILL, CFILL, CIRC, GREY = "#1a1a1a", "#DCE9F5", "#FDEBD3", "#F8E3DA", "#B4441A", "#444444"
FT, FB, LS, TOP = 8.2, 7.2, 0.135, 2.95


def box(x, y, w, h, title, lines=(), fill=FILL, lw=0.8, ec=INK):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=ec, lw=lw))
    rows = [(title, True)] + [(t, False) for t in lines]
    yy = y + h / 2 + (len(rows) - 1) * LS / 2
    for t, b in rows:
        ax.text(x + w / 2, yy, t, ha="center", va="center", fontsize=FT if b else FB,
                weight="bold" if b else "normal", color=INK)
        yy -= LS


def arrow(p, q, color=INK, dashed=False, lw=0.85):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7.5, color=color, lw=lw,
                                 linestyle=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0))


def head(x, w, t):
    ax.text(x + w / 2, TOP + 0.20, t, ha="center", va="center", fontsize=FT, weight="bold", color=GREY)
    ax.plot([x, x + w], [TOP + 0.09, TOP + 0.09], color=GREY, lw=0.6)

ax_, aw = 0.05, 1.30
head(ax_, aw, "(a) Knowledge")
box(ax_, 2.30, aw, 0.62, "Lecture slides", ["37 decks,", "2 computing courses"])
box(ax_, 1.43, aw, 0.62, "KB construction", ["5 methods (rule-based,", "LLM, entailment-gated)"])
box(ax_, 0.56, aw, 0.62, "Knowledge base", ["facts and their", "rule-made negations"])
arrow((ax_ + aw / 2, 2.30), (ax_ + aw / 2, 2.05))
arrow((ax_ + aw / 2, 1.43), (ax_ + aw / 2, 1.18))

bx, bw = 1.62, 1.62
head(bx, bw, "(b) Test statements")
box(bx, 2.25, bw, 0.67, "Circular errors", ["made from the KB:", "negations, LLM edits"], fill=CFILL, ec=CIRC, lw=1.0)
box(bx, 1.35, bw, 0.75, "Independent errors", ["not from the KB: 150 course,", "85 ProgMiscon (expert),", "149 FEVER refuted"])
box(bx, 0.45, bw, 0.75, "Correct statements", ["matched to the errors:", "136 course, 85 ProgMiscon,", "148 FEVER supported"])
arrow((ax_ + aw, 0.95), (bx, 2.42), color=CIRC, dashed=True, lw=1.1)

cx, cw = 3.62, 1.36
head(cx, cw, "(c) Guards under test")
gy0, gy1 = 0.56, 2.92
ax.add_patch(Rectangle((cx, gy0), cw, gy1 - gy0, fc=GFILL, ec=INK, lw=0.8))


def ctext(y, t, b=False):
    ax.text(cx + cw / 2, y, t, ha="center", va="center", fontsize=FT if b else FB,
            weight="bold" if b else "normal", color=INK)

ctext(2.70, "KBGuard", True)
items = ["claim splitting", "retrieval from the KB", "NLI checker (one of 6)", "decision rule"]
for k, t in enumerate(items):
    ctext(2.47 - k * 0.27, t)
    if k < 3:
        ax.add_patch(FancyArrowPatch((cx + cw / 2, 2.47 - k * 0.27 - 0.075), (cx + cw / 2, 2.47 - (k + 1) * 0.27 + 0.075),
                                     arrowstyle="-|>", mutation_scale=5, color=GREY, lw=0.6, shrinkA=0, shrinkB=0))
ax.plot([cx + 0.08, cx + cw - 0.08], [1.36, 1.36], color=GREY, lw=0.6, linestyle=(0, (3, 2)))
ctext(1.12, "Published guards", True)
ctext(0.95, "MiniCheck, HHEM")
ctext(0.82, "(KB facts as document)")
bus = bx + bw + 0.14
for y in (2.585, 1.725, 0.825):
    ax.plot([bx + bw, bus], [y, y], color=INK, lw=0.85)
ax.plot([bus, bus], [0.825, 2.585], color=INK, lw=0.85)
arrow((bus, 1.74), (cx, 1.74))
ax.plot([ax_ + aw / 2, ax_ + aw / 2], [0.56, 0.18], color=INK, lw=0.85)
ax.plot([ax_ + aw / 2, cx + cw / 2], [0.18, 0.18], color=INK, lw=0.85)
arrow((cx + cw / 2, 0.18), (cx + cw / 2, gy0))
ax.text(1.0, 0.26, "same KB", ha="center", va="center", fontsize=FB, style="italic")

dx, dw = 5.30, 1.38
head(dx, dw, "(d) Measures")
box(dx, 2.36, dw, 0.56, "Catch rate", ["errors flagged"])
box(dx, 1.68, dw, 0.56, "FCR", ["correct statements flagged"])
box(dx, 1.00, dw, 0.56, "AUROC, C@5%", ["threshold-free separation"])
box(dx, 0.32, dw, 0.56, "Circular gap", ["circular minus independent"])
bus2 = dx - 0.14
ax.plot([cx + cw, bus2], [1.74, 1.74], color=INK, lw=0.85)
ax.plot([bus2, bus2], [0.60, 2.64], color=INK, lw=0.85)
for y in (2.64, 1.96, 1.28, 0.60):
    arrow((bus2, y), (dx, y))

ax.plot([3.25, 3.55], [0.08, 0.08], color=INK, lw=0.85)
ax.text(3.60, 0.08, "data flow", va="center", fontsize=FB)
ax.plot([4.25, 4.55], [0.08, 0.08], color=CIRC, lw=1.1, linestyle=(0, (4, 2.5)))
ax.text(4.60, 0.08, "circular link (errors made from the KB)", va="center", fontsize=FB)

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "Figure_1.pdf"), bbox_inches="tight", pad_inches=0.02)
print("wrote", os.path.join(out, "Figure_1.pdf"))
