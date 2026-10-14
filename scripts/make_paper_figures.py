"""All result figures of the journal version, in Elsevier's artwork format:
widths of 90 mm (single column) or 140 mm (1.5 columns), named by figure number
(Figure_2, Figure_4, Figure_5, Figure_6, Figure_A1), Arial-compatible font
(Liberation Sans), every label at least 7 pt at final size, vector PDF with embedded fonts.

  python3 scripts/make_paper_figures.py exp/results/accepted/summary.json figures

GPT-6 Sol, ProgMiscon and oracle values are the reported results
(exp/results/accepted/summary.json, exp/results/progmiscon/paired.json).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
W1, W15 = 90 * MM, 140 * MM
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.6,
                     "axes.linewidth": 0.6})
INK, MUTED, GRID = "#2b2b2b", "#8a8a85", "#e6e6e3"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
LBL = 7          # smallest text anywhere


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, width=0.6)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def bar_labels(ax, xs, vs, dy=1.5):
    for x, v in zip(xs, vs):
        ax.text(x, v + dy, f"{v:.0f}", ha="center", va="bottom", fontsize=LBL, color=INK)


g = json.load(open(sys.argv[1]))["groups"]
out = sys.argv[2]
R = "roberta-large-mnli"

# ---------------------------------------------------------------- circular vs independent, by path
paths = [("all", "All paths"), ("nli_fact+nli_neg", "(a)+(b)"), ("nli_fact_only", "(a) only")]
fig, axes = plt.subplots(1, 2, figsize=(W15, 58 * MM), sharey=True)
for ax, (arm, title) in zip(axes, [("baseline_rule", "Rule-based KB"), ("kca", "KCA KB")]):
    m = g[f"{arm}|{R}|concept"]["circular_matched_by_paths"]
    non = [m[p]["noncircular"] for p, _ in paths]; cir = [m[p]["circular"] for p, _ in paths]
    x = np.arange(3); w = .38
    ax.bar(x - w / 2 - .01, non, w, color=BLUE, label="Independent misconceptions")
    ax.bar(x + w / 2 + .01, cir, w, color=ORANGE, hatch="///", edgecolor="white", lw=0,
           label="Circular errors (stored negations)")
    bar_labels(ax, x - w / 2, non); bar_labels(ax, x + w / 2, cir)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in paths])
    ax.set_title(title, fontsize=7.5, color=INK); ax.set_ylim(0, 112); style(ax)
axes[0].set_ylabel("Catch rate, same concepts (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(f"{out}/Figure_2.pdf"); plt.close(fig)

# ---------------------------------------------------------------- contradiction signal by fact length
bands = ["<=8 words", "9-15", "16-30", ">30"]
fig, ax = plt.subplots(figsize=(W1, 55 * MM))
for arm, lab, col, mk in [("baseline_rule", "Rule-based", BLUE, "o"), ("kca", "KCA", ORANGE, "s")]:
    m = g[f"{arm}|{R}|concept"]["mechanism"]["mean_contra_by_length"]
    ax.plot(range(4), [m[b] for b in bands], color=col, lw=1.6, marker=mk, ms=5, mec="white", label=lab)
ax.set_xticks(range(4)); ax.set_xticklabels(["≤8", "9–15", "16–30", ">30"])
ax.set_xlabel("Stored fact length (words)"); ax.set_ylabel("Mean contradiction probability")
style(ax); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{out}/Figure_A1.pdf"); plt.close(fig)

# ---------------------------------------------------------------- checkers on the KCA KB
V = [("roberta-large-mnli", "RoBERTa"), ("DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "DeBERTa"),
     ("ModernBERT-large-nli", "Modern-\nBERT"), ("llm:deepseek/deepseek-v4.1-flash", "DeepSeek"),
     ("llm:qwen/qwen3.8-flash", "Qwen"), ("llm:openai/gpt-6-sol", "GPT-6\nSol")]
cir, non, fcr, lab = [], [], [], []
for k, l in V:
    d = g.get(f"kca|{k}|concept")
    if d is None and k == "llm:openai/gpt-6-sol":       # reported by the frontier run
        cir.append(98.2); non.append(83.9); fcr.append(11.0); lab.append(l); continue
    m = d["circular_matched_by_paths"]["all"]
    cir.append(m["circular"]); non.append(m["noncircular"]); fcr.append(d["fcr"]["rate"]); lab.append(l)
x = np.arange(len(lab)); w = .27
fig, ax = plt.subplots(figsize=(W15, 62 * MM))
ax.bar(x - w - .02, cir, w, color=ORANGE, hatch="///", edgecolor="white", lw=0, label="Circular errors")
ax.bar(x, non, w, color=BLUE, label="Real misconceptions (same concepts)")
ax.bar(x + w + .02, fcr, w, color=GRAY, label="False alarms on correct statements")
for xs, vs in [(x - w - .02, cir), (x, non), (x + w + .02, fcr)]:
    bar_labels(ax, xs, vs)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("Rate (%)"); ax.set_ylim(0, 114); style(ax)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(f"{out}/Figure_4.pdf"); plt.close(fig)

# ---------------------------------------------------------------- ProgMiscon: catch vs false alarms
P = [("RoBERTa (ours)", 48.2, 50.6, "nli"), ("MiniCheck", 65.9, 62.4, "nli"), ("HHEM", 55.3, 50.6, "nli"),
     ("Qwen", 15.3, 29.4, "llm"), ("GPT-6 Sol", 16.5, 47.1, "llm")]
fig, ax = plt.subplots(figsize=(W1, 82 * MM))
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1, label="Chance (catch = false alarms)")
for name, f, c, kind in P:
    col, mk = (BLUE, "o") if kind == "nli" else (AQUA, "s")
    ax.scatter(f, c, s=34, color=col, marker=mk, edgecolor="white", linewidth=1.2, zorder=3)
off = {"RoBERTa (ours)": (-2, 4, "right"), "MiniCheck": (3, -1.5, "left"), "HHEM": (3, -6, "left"),
       "Qwen": (3, -4, "left"), "GPT-6 Sol": (3, -5, "left")}
for name, f, c, _ in P:
    dx, dy, ha = off[name]
    ax.text(f + dx, c + dy, name, fontsize=LBL, color=INK, ha=ha)
ax.scatter([], [], color=BLUE, marker="o", s=24, label="NLI-based checkers")
ax.scatter([], [], color=AQUA, marker="s", s=24, label="LLM checkers")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
ax.set_xlabel("False alarms on the corrections (%)"); ax.set_ylabel("Catch rate on the misconceptions (%)")
ax.xaxis.grid(True, color=GRID, lw=0.6); style(ax)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{out}/Figure_5.pdf"); plt.close(fig)

# ---------------------------------------------------------------- oracle KB: wording vs coverage
O = [("RoBERTa\n(ours)", 100.0, 78.8, 50.6), ("Qwen", 100.0, 90.6, 29.4), ("GPT-6\nSol", 100.0, 92.9, 47.1),
     ("MiniCheck", 97.6, 92.9, 62.4), ("HHEM", 42.4, 32.9, 50.6)]
x = np.arange(len(O)); w = .27
fig, ax = plt.subplots(figsize=(W15, 62 * MM))
t = [o[1] for o in O]; e = [o[2] for o in O]; c = [o[3] for o in O]
ax.bar(x - w - .02, t, w, color=ORANGE, hatch="///", edgecolor="white", lw=0, label="Template negation, oracle KB")
ax.bar(x, e, w, color=BLUE, label="Expert misconception, oracle KB")
ax.bar(x + w + .02, c, w, color=AQUA, label="Expert misconception, course KB")
for xs, vs in [(x - w - .02, t), (x, e), (x + w + .02, c)]:
    bar_labels(ax, xs, vs)
ax.set_xticks(x); ax.set_xticklabels([o[0] for o in O]); ax.set_ylabel("Catch rate (%)"); ax.set_ylim(0, 114)
style(ax)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, .09, 1, 1)); fig.savefig(f"{out}/Figure_6.pdf"); plt.close(fig)
print("ok")
