"""make_overview_figure.py -- Figure 1: study overview.

Plain academic style of Computers & Education figures (square pastel boxes, bold
upper-case titles, stage blocks with header bars, one dark-teal arrow colour, legend for
solid vs dashed arrows), with simple line icons drawn from basic shapes (no clip art,
no third-party icon sets, so no licence questions).
USAGE: python3 scripts/make_overview_figure.py figures   (writes figures/Figure_1.pdf)
Drawn at print size (5.4 in = text width); vector PDF with embedded TrueType fonts."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, FancyArrowPatch, FancyBboxPatch, Circle, Ellipse,
                                Wedge, Polygon)

plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})
FT, FS, FL = 7.4, 6.9, 6.6           # box title, body, label (pt)
LS = 0.128                           # line spacing (in)

W, H = 5.4, 4.45
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK, TEAL = "#1a1a1a", "#1F5F7A"
BLUE, PINK, GREEN, PEACH, GOLD, BODY = "#CDE8F6", "#F2D3EE", "#CDEFD2", "#FBE1D3", "#F6C343", "#DCEFF9"
OKGREEN, FLAGRED = "#2E8B57", "#C8501A"
IC = TEAL                            # icon colour


# ---------------------------------------------------------------- icons
def icon_student(cx, cy, s=0.11):
    ax.add_patch(Wedge((cx, cy - 1.0 * s), 0.95 * s, 0, 180, fc=IC, ec="none"))
    ax.add_patch(Circle((cx, cy + 0.12 * s), 0.42 * s, fc=IC, ec="white", lw=0.6))
    ax.add_patch(Polygon([(cx - 0.75 * s, cy + 0.62 * s), (cx, cy + 0.92 * s), (cx + 0.75 * s, cy + 0.62 * s),
                          (cx, cy + 0.36 * s)], closed=True, fc=INK, ec="none"))      # mortarboard
    ax.plot([cx + 0.6 * s, cx + 0.6 * s], [cy + 0.62 * s, cy + 0.25 * s], color=INK, lw=0.6)


def icon_chip(cx, cy, s=0.11):
    for k in (-0.45, 0, 0.45):
        for (dx, dy, ex, ey) in ((k, 0.62, k, 0.95), (k, -0.62, k, -0.95), (0.62, k, 0.95, k), (-0.62, k, -0.95, k)):
            ax.plot([cx + dx * s, cx + ex * s], [cy + dy * s, cy + ey * s], color=IC, lw=0.9, solid_capstyle="butt")
    ax.add_patch(FancyBboxPatch((cx - 0.65 * s, cy - 0.65 * s), 1.3 * s, 1.3 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.18 * s}", fc=IC, ec="none"))
    ax.text(cx, cy, "AI", ha="center", va="center", fontsize=4.6, weight="bold", color="white")


def icon_shield(cx, cy, s=0.12):
    pts = [(cx, cy + 0.95 * s), (cx + 0.78 * s, cy + 0.65 * s), (cx + 0.72 * s, cy - 0.1 * s),
           (cx, cy - 0.98 * s), (cx - 0.72 * s, cy - 0.1 * s), (cx - 0.78 * s, cy + 0.65 * s)]
    ax.add_patch(Polygon(pts, closed=True, fc=IC, ec="none"))
    ax.plot([cx - 0.32 * s, cx - 0.06 * s, cx + 0.38 * s], [cy + 0.02 * s, cy - 0.26 * s, cy + 0.36 * s],
            color="white", lw=1.2, solid_capstyle="round", solid_joinstyle="round")


def icon_slides(cx, cy, s=0.11):
    for d, fc in ((0.22, "white"), (0.0, "white")):
        x0, y0 = cx - 0.8 * s + d * s, cy - 0.6 * s + d * s
        ax.add_patch(Rectangle((x0, y0), 1.4 * s, 1.05 * s, fc=fc, ec=IC, lw=0.8))
    x0, y0 = cx - 0.8 * s, cy - 0.6 * s
    ax.add_patch(Rectangle((x0, y0 + 0.78 * s), 1.4 * s, 0.27 * s, fc=IC, ec="none"))
    for i, yy in enumerate((0.55, 0.33, 0.11)):
        ax.add_patch(Circle((x0 + 0.2 * s, y0 + yy * s), 0.05 * s, fc=IC, ec="none"))
        ax.plot([x0 + 0.35 * s, x0 + (1.15 - 0.2 * i) * s], [y0 + yy * s] * 2, color=IC, lw=0.7)


def icon_db(cx, cy, s=0.11):
    w, h, e = 1.3 * s, 1.35 * s, 0.38 * s
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=IC, ec="none"))
    ax.add_patch(Ellipse((cx, cy - h / 2), w, e, fc=IC, ec="none"))
    for yy in (cy - h / 2 + h / 3, cy - h / 2 + 2 * h / 3):
        ax.add_patch(Ellipse((cx, yy), w, e, fc="none", ec="white", lw=0.7, theta1=180, theta2=360) if False else
                     matplotlib.patches.Arc((cx, yy), w, e, theta1=180, theta2=360, ec="white", lw=0.7))
    ax.add_patch(Ellipse((cx, cy + h / 2), w, e, fc="#5C93AA", ec=IC, lw=0.6))


def icon_ok(cx, cy, s=0.10):
    ax.add_patch(Circle((cx, cy), s, fc=OKGREEN, ec="none"))
    ax.plot([cx - 0.45 * s, cx - 0.1 * s, cx + 0.5 * s], [cy + 0.0 * s, cy - 0.38 * s, cy + 0.4 * s],
            color="white", lw=1.2, solid_capstyle="round", solid_joinstyle="round")


def icon_flag(cx, cy, s=0.10):
    ax.add_patch(Circle((cx, cy), s, fc=FLAGRED, ec="none"))
    ax.plot([cx, cx], [cy - 0.05 * s, cy + 0.55 * s], color="white", lw=1.3, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy - 0.42 * s), 0.11 * s, fc="white", ec="none"))


def icon_list(cx, cy, s=0.075):
    for yy in (0.6, 0, -0.6):
        ax.add_patch(Circle((cx - 0.75 * s, cy + yy * s), 0.17 * s, fc=INK, ec="none"))
        ax.plot([cx - 0.4 * s, cx + 0.9 * s], [cy + yy * s] * 2, color=INK, lw=1.0)


def icon_gear(cx, cy, s=0.075):
    import numpy as np
    n, pts = 8, []
    for i in range(4 * n):
        a = 2 * np.pi * i / (4 * n)
        r = 1.0 if (i % 4) in (0, 1) else 0.74
        pts.append((cx + r * s * np.cos(a), cy + r * s * np.sin(a)))
    ax.add_patch(Polygon(pts, closed=True, fc=INK, ec="none"))
    ax.add_patch(Circle((cx, cy), 0.32 * s, fc=GOLD, ec="none"))


def icon_bars(cx, cy, s=0.075):
    for i, hgt in enumerate((0.7, 1.2, 1.7)):
        ax.add_patch(Rectangle((cx - 0.9 * s + i * 0.65 * s, cy - 0.85 * s), 0.45 * s, hgt * s, fc=INK, ec="none"))
    ax.plot([cx - 1.0 * s, cx + 1.0 * s], [cy - 0.85 * s] * 2, color=INK, lw=0.7)


# ---------------------------------------------------------------- helpers
def box(x, y, w, h, title, sub=(), fill="white", icon=None, top_pad=0.0):
    """box with an optional icon centred above the text"""
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=INK, lw=0.8))
    rows = [(t, True) for t in ([title] if isinstance(title, str) else title)] + [(t, False) for t in sub]
    text_h = len(rows) * LS
    if icon:
        icon(x + w / 2, y + h - 0.20 - top_pad)
        yy = y + (h - 0.36 - top_pad) / 2 + (len(rows) - 1) * LS / 2
    else:
        yy = y + h / 2 + (len(rows) - 1) * LS / 2
    for t, bold in rows:
        ax.text(x + w / 2, yy, t, ha="center", va="center", fontsize=FT if bold else FS,
                weight="bold" if bold else "normal", color=INK)
        yy -= LS


def side_box(x, y, w, h, title, sub, fill, icon):
    """box with the icon on the left (used for the two outcomes)"""
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=INK, lw=0.8))
    icon(x + 0.155, y + h / 2, s=0.085)
    ax.text(x + 0.28, y + h / 2 + LS / 2, title, ha="left", va="center", fontsize=6.9, weight="bold", color=INK)
    ax.text(x + 0.28, y + h / 2 - LS / 2, sub, ha="left", va="center", fontsize=FS, color=INK)


def arrow(p, q, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=TEAL, lw=1.1,
                                 ls=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def lab(x, y, t, ha="center"):
    ax.text(x, y, t, ha=ha, va="center", fontsize=FL, color=INK)


def stage(x, y, w, h, head, bullets, icon):
    hh = 0.26
    ax.add_patch(Rectangle((x, y + h - hh), w, hh, fc=GOLD, ec="none"))
    ax.add_patch(Rectangle((x, y), w, h - hh, fc=BODY, ec="none"))
    icon(x + 0.20, y + h - hh / 2)
    ax.text(x + w / 2 + 0.10, y + h - hh / 2, head, ha="center", va="center", fontsize=FT + 0.3,
            weight="bold", color=INK)
    yy = y + h - hh - 0.13
    for b in bullets:
        first = True
        for line in b.split("\n"):
            if first:
                ax.text(x + 0.07, yy, "•", ha="left", va="center", fontsize=FS, color=INK)
            ax.text(x + 0.19, yy, line, ha="left", va="center", fontsize=FS, color=INK)
            yy -= LS; first = False
        yy -= 0.03


# ---------------- (a) the guard inside an AI tutor ----------------
ax.text(0.03, 4.34, "(a)", fontsize=8.5, weight="bold", color=INK, va="center")
yr, hr = 3.30, 0.78
box(0.20, yr, 0.95, hr, "STUDENT", ["asks a question"], BLUE, icon_student)
box(1.40, yr, 0.95, hr, "TUTOR LLM", ["drafts an answer"], BLUE, icon_chip)
box(2.60, yr - 0.12, 1.30, hr + 0.24, ["HALLUCINATION", "GUARD"], ["checks each claim", "against the KB"],
    PINK, icon_shield, top_pad=0.01)
side_box(4.10, yr + 0.47, 1.28, 0.46, "ANSWER SHOWN", "no conflict", GREEN, icon_ok)
side_box(4.10, yr - 0.15, 1.28, 0.46, "ANSWER FLAGGED", "conflict found", PEACH, icon_flag)
my = yr + hr / 2
arrow((1.15, my), (1.40, my)); arrow((2.35, my), (2.60, my))
arrow((3.90, my + 0.14), (4.10, yr + 0.70)); arrow((3.90, my - 0.14), (4.10, yr + 0.08))

yk = 2.30
box(0.20, yk, 1.25, 0.66, "LECTURE SLIDES", ["37 decks, 2 courses"], GREEN, icon_slides)
box(1.80, yk, 2.10, 0.66, "KNOWLEDGE BASE (KB)", ["facts + one rule-made negation each"], GREEN, icon_db)
arrow((1.45, yk + 0.33), (1.80, yk + 0.33)); lab(1.625, yk + 0.45, "5 ways")
arrow((3.25, yk + 0.66), (3.25, yr - 0.12)); lab(3.27, yk + 0.77, "  evidence", ha="left")

# ---------------- (b) evaluation design ----------------
ax.text(0.03, 1.72, "(b)", fontsize=8.5, weight="bold", color=INK, va="center")
sy, sh, sw = 0.03, 1.50, 1.62
xs = [0.20, 1.94, 3.68]
stage(xs[0], sy, sw, sh, "1  Test statements",
      ["Circular errors, made from\nthe KB: rule-made negations;\nLLM-written false edits",
       "Independent errors: 150 course\nmisconceptions, 85 ProgMiscon,\n149 FEVER refuted",
       "Correct statements: 136 course,\n85 ProgMiscon, 148 FEVER"], icon_list)
stage(xs[1], sy, sw, sh, "2  Systems under test",
      ["Our guard with 6 checkers,\non 5 course KBs and FEVER",
       "Published guards:\nMiniCheck, HHEM",
       "Chunk verifier (Appendix B)",
       "Controls: path (a) only, oracle\nKB, open retrieval, held-out\nitems"], icon_gear)
stage(xs[2], sy, sw, sh, "3  Measures",
      ["Catch rate on errors",
       "False-contradiction rate (FCR)\non correct statements",
       "AUROC; catch at FCR ≤ 5%",
       "Circular gap: circular minus\nindependent catch rate on the\nsame concepts (McNemar)"], icon_bars)
for x in xs[:2]:
    ax.plot([x + sw, x + sw + 0.12], [sy + sh - 0.13] * 2, color=TEAL, lw=1.6)

# circularity: test errors are made from the KB itself
arrow((1.95, yk), (1.10, sy + sh), dashed=True, rad=0.15)
lab(1.70, 1.92, "circular errors are made", ha="left"); lab(1.70, 1.80, "from the KB's own facts", ha="left")

# legend
ax.plot([4.05, 4.40], [1.97, 1.97], color=TEAL, lw=1.1)
lab(4.45, 1.97, ": system flow", ha="left")
ax.plot([4.05, 4.40], [1.81, 1.81], color=TEAL, lw=1.1, ls=(0, (4, 2.5)))
lab(4.45, 1.81, ": circular link", ha="left")

out = sys.argv[1] if len(sys.argv) > 1 else "Figure_1.pdf"
if os.path.isdir(out):
    out = os.path.join(out, "Figure_1.pdf")
fig.savefig(out)
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=250)
print("wrote", out)
