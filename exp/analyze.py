"""
analyze.py -- every number, table and figure for the paper, from exp/runs/*.jsonl.

Nothing here runs a model; it replays logged scores (exp/replay.py).

Per run group (KB arm x NLI model x retrieval mode) it reports
  catch rate (fixed gate, legacy gate, no gate) with Wilson 95% CIs
  false-contradiction rate (FCR) on the true-statement set
  false-verification rate: errors the guard labels VERIFIED
  AUROC of the verifier's contradiction score, errors vs true statements
  catch at FCR <= 5%
  circular catch rate on the same concepts, and the paired inflation
  path attribution (nli_fact / nli_neg / cosine) and path-ablated catch rates
Across arms (same model, same retrieval): exact McNemar vs the baseline arm.
Mechanism: fact length vs NLI contradiction probability over all logged pairs.
Self-recognition: catch rate by lexical-overlap tercile and the between/within
decomposition of the circular-vs-non-circular gap.

Item subsets: all | heldout (not in the 55-item development set) |
scope (concept present in every KB of the comparison) | dedup (one item per
near-duplicate cluster).

USAGE:  python3 exp/analyze.py --runs exp/runs --out exp/results
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from exp.replay import item_verdict, DEFAULTS  # noqa: E402

PATH_SETS = {"all": ("nli_fact", "nli_neg", "cosine"),
             "nli_fact+nli_neg": ("nli_fact", "nli_neg"),
             "nli_fact_only": ("nli_fact",)}


# ------------------------------------------------------------------ utils
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def pct(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * x, 1)


def load_runs(run_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        if not lines or not lines[0].get("_meta"):
            continue
        meta, recs = lines[0], lines[1:]
        kinds = Counter(r.get("kind") for r in recs)
        meta["set"] = kinds.most_common(1)[0][0] if kinds else "?"
        meta["arm"] = os.path.basename(meta["kb"]).replace(".jsonl", "") if not meta["kb"].startswith("neo4j:") \
            else "orig_" + meta["kb"].split(":", 1)[1]
        m = meta["nli_model"]
        meta["model"] = m.split("@")[0] if m.startswith("llm:") else m.split("/")[-1]
        meta["path"] = p
        runs.append((meta, recs))
    return runs


def accepted_true_ids(path=None):
    """IDs of the course true statements both annotators accepted."""
    path = path or os.path.join(ROOT, "exp", "data", "true_statements.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run kappa_true_statements.py first")
    return {it["id"] for it in json.load(open(path, encoding="utf-8")) if it.get("review") == "kept"}


def filter_accepted(runs):
    keep = accepted_true_ids()
    out = []
    for meta, recs in runs:
        if meta.get("set") == "true" and not meta["arm"].startswith("fever_"):
            recs = [r for r in recs if r.get("id") in keep]
        out.append((meta, recs))
    print(f"accepted-only: course true statements restricted to {len(keep)} items")
    return out


def item_score(rec, floor=DEFAULTS["relevance_floor"], gate=True):
    """Continuous contradiction score used for AUROC: the guard's strongest NLI
    contradiction signal that beats its support signal and passes the fixed gate."""
    best = 0.0
    for c in rec.get("claims", []):
        fc, cn, sup, fact = 0.0, 0.0, 0.0, None
        for q in c["pairs"]:
            if q["contra"] > fc:
                fc, fact = q["contra"], q["fact_id"]
            sup = max(sup, q["entail"])
            if q["neg_entail"] is not None and q["neg_entail"] > cn:
                cn = q["neg_entail"]
                if fact is None:
                    fact = q["fact_id"]
        s = max(fc, cn)
        if s <= sup:
            continue
        if gate and fact and rec["facts"].get(fact, {}).get("gate_sim", 1.0) < floor:
            continue
        best = max(best, s)
    return best


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def catch_at_fcr(pos, neg, max_fcr=0.05):
    best = 0.0
    for t in sorted(set(neg) | set(pos) | {1.01}):
        fcr = np.mean([s >= t for s in neg]) if neg else 0
        if fcr <= max_fcr:
            best = max(best, float(np.mean([s >= t for s in pos])))
    return best


# --------------------------------------------------------------- analysis
def flags(recs, gate="fixed", paths=PATH_SETS["all"], **kw):
    out = {}
    for r in recs:
        if r.get("online") == "NO_FACTS":
            out[r["id"]] = {"raw": "NO_FACTS", "gated": "NO_FACTS", "path": None}
        else:
            out[r["id"]] = item_verdict(r, gate=gate, paths=paths, **kw)
    return out


def subset_ids(recs, which, scope_concepts=None):
    ids = []
    seen_cluster = set()
    for r in recs:
        if which == "heldout" and r.get("in_dev55"):
            continue
        if which == "scope" and scope_concepts is not None and r["concept"] not in scope_concepts:
            continue
        if which == "dedup":
            cl = r.get("dup_cluster")
            if cl in seen_cluster:
                continue
            seen_cluster.add(cl)
        ids.append(r["id"])
    return ids


def rate(fl, ids, field="gated", value="CONTRADICTED"):
    k = sum(1 for i in ids if fl[i][field] == value)
    n = len(ids)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": pct(k / n) if n else None, "ci": [pct(lo), pct(hi)]}


def tokens_overlap_terciles(err_recs, circ_recs, err_fl, circ_fl):
    def ov(r):
        return max([f.get("jaccard", 0) for f in r["facts"].values()] or [0])
    e = [(ov(r), err_fl[r["id"]]["gated"] == "CONTRADICTED") for r in err_recs if r["facts"]]
    c = [(ov(r), circ_fl[r["id"]]["gated"] == "CONTRADICTED") for r in circ_recs if r["facts"]]
    if not e or not c:
        return None
    if len(e) < 6 or len(c) < 6:          # too few items for terciles; report means only
        return {"mean_overlap_noncirc": round(float(np.mean([v for v, _ in e])), 3),
                "mean_overlap_circ": round(float(np.mean([v for v, _ in c])), 3)}
    allv = [v for v, _ in e + c]
    q1, q2 = np.quantile(allv, [1 / 3, 2 / 3])
    b = lambda v: 0 if v <= q1 else (1 if v <= q2 else 2)
    out = {"cuts": [round(float(q1), 3), round(float(q2), 3)], "bins": []}
    pi_e, pi_c, p_e, p_c = [], [], [], []
    for k in range(3):
        ek = [x for v, x in e if b(v) == k]
        ck = [x for v, x in c if b(v) == k]
        pi_e.append(len(ek) / len(e)); pi_c.append(len(ck) / len(c))
        p_e.append(np.mean(ek) if ek else 0.0); p_c.append(np.mean(ck) if ck else 0.0)
        out["bins"].append({"bin": ["low", "mid", "high"][k], "n_noncirc": len(ek), "n_circ": len(ck),
                            "catch_noncirc": pct(np.mean(ek)) if ek else None,
                            "catch_circ": pct(np.mean(ck)) if ck else None})
    gap = sum(pi_c[k] * p_c[k] for k in range(3)) - sum(pi_e[k] * p_e[k] for k in range(3))
    between = sum((pi_c[k] - pi_e[k]) * p_e[k] for k in range(3))
    out.update(mean_overlap_noncirc=round(float(np.mean([v for v, _ in e])), 3),
               mean_overlap_circ=round(float(np.mean([v for v, _ in c])), 3),
               gap_points=pct(gap), between_bin_points=pct(between), within_bin_points=pct(gap - between))
    return out


def mechanism(err_recs, floor=DEFAULTS["relevance_floor"]):
    words, contra, ontopic_w, ontopic_c = [], [], [], []
    for r in err_recs:
        for c in r.get("claims", []):
            for q in c["pairs"]:
                f = r["facts"].get(q["fact_id"])
                if not f:
                    continue
                words.append(f["words"]); contra.append(q["contra"])
                if f.get("gate_sim", 0) >= floor:
                    ontopic_w.append(f["words"]); ontopic_c.append(q["contra"])
    res = {"n_pairs": len(words)}
    if len(words) > 10:
        rho, p = stats.spearmanr(words, contra)
        res.update(spearman_all=round(float(rho), 3), p_all=float(p))
    if len(ontopic_w) > 10:
        rho, p = stats.spearmanr(ontopic_w, ontopic_c)
        res.update(spearman_ontopic=round(float(rho), 3), p_ontopic=float(p), n_ontopic=len(ontopic_w))
        w = np.array(ontopic_w); cc = np.array(ontopic_c)
        res["mean_contra_by_length"] = {lab: (round(float(cc[m].mean()), 3) if m.any() else None)
                                        for lab, m in (("<=8 words", w <= 8), ("9-15", (w > 8) & (w <= 15)),
                                                       ("16-30", (w > 15) & (w <= 30)), (">30", w > 30))}
    return res


def dup_share(recs):
    """Share of claims whose five NLI slots contain two or more identical fact texts."""
    import re as _re
    n = d = 0
    for r in recs:
        for c in r.get("claims", []):
            texts = [_re.sub(r"\s+", " ", r["facts"].get(q["fact_id"], {}).get("claim", "").lower().strip())
                     for q in c["pairs"]]
            texts = [t for t in texts if t]
            if not texts:
                continue
            n += 1
            d += len(texts) != len(set(texts))
    return pct(d / n) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "exp", "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "results"))
    ap.add_argument("--pairs", nargs="+", default=[
        "kca:baseline_rule", "sentence_split:baseline_rule", "llm_only:baseline_rule",
        "kca:llm_only", "kca:sentence_split",
        "kca_regate_minicheck_flan_t5_large:kca",
        "kca_regate_deberta_v3_large_mnli_fever_anli_ling_wanli:kca",
        "fever_kca:fever_fragment", "fever_sentence:fever_fragment", "fever_kca:fever_sentence",
        "fever_kca:fever_llm_only"],
        help="arm:reference pairs compared item-by-item")
    ap.add_argument("--accepted-only", action="store_true",
                    help="keep only the course true statements both annotators accepted "
                         "(review == 'kept' in exp/data/true_statements.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.runs)
    if not runs:
        sys.exit(f"No runs found in {args.runs}")
    if args.accepted_only:
        runs = filter_accepted(runs)
    groups = defaultdict(dict)
    for meta, recs in runs:
        key = (meta["arm"], meta["model"], meta["retrieval"] + ("" if meta.get("expand", True) else "-noexp"))
        groups[key][meta["set"]] = (meta, recs)

    # scope = concepts present in every KB arm that was run (per model+retrieval)
    # (computed separately for the course testbed and for FEVER, whose concepts differ)
    family = lambda arm: "fever" if arm.startswith("fever_") else "course"
    present = defaultdict(list)
    for (arm, model, retr), sets in groups.items():
        if "error" in sets:
            present[(family(arm), model, retr)].append(
                {r["concept"] for r in sets["error"][1] if r.get("n_concept_facts", 0) > 0})
    scope_f = {k: set.intersection(*v) if v else set() for k, v in present.items()}

    class _Scope(dict):
        def get(self, key, default=None):     # key is (model, retr); arm comes from the loop
            return scope_f.get((self.fam,) + tuple(key), default)
    scope = _Scope()

    summary = {"groups": {}, "paired": {}, "notes": []}
    for (arm, model, retr), sets in sorted(groups.items()):
        scope.fam = family(arm)
        g = {}
        if "error" in sets:
            recs = sets["error"][1]
            fl = {gate: flags(recs, gate=gate) for gate in ("fixed", "legacy", "none")}
            for sub in ("all", "heldout", "scope", "dedup"):
                ids = subset_ids(recs, sub, scope.get((model, retr)))
                g[f"catch_{sub}"] = rate(fl["fixed"], ids)
            ids_all = subset_ids(recs, "all")
            g["catch_legacy_gate"] = rate(fl["legacy"], ids_all)
            g["catch_no_gate"] = rate(fl["none"], ids_all)
            g["false_verified_on_errors"] = rate(fl["fixed"], ids_all, field="raw", value="VERIFIED")
            g["no_facts"] = sum(1 for r in recs if r.get("online") == "NO_FACTS")
            g["fallback_retrieval"] = sum(1 for r in recs if r.get("fallback"))
            g["replay_mismatches"] = sum(1 for r in recs if "replay_mismatch" in r)
            g["path_of_catches"] = dict(Counter(fl["fixed"][i]["path"] for i in ids_all
                                                if fl["fixed"][i]["gated"] == "CONTRADICTED"))
            for name, ps in PATH_SETS.items():
                g[f"catch_paths_{name}"] = rate(flags(recs, paths=ps), ids_all)
            g["mechanism"] = mechanism(recs)
            g["mechanism"]["claims_with_duplicate_in_top5"] = dup_share(recs)
            err_scores = [item_score(r) for r in recs]
        if "true" in sets:
            trecs = sets["true"][1]
            tfl = flags(trecs)
            tids = subset_ids(trecs, "all")
            g["fcr"] = rate(tfl, tids)
            g["fcr_scope"] = rate(tfl, subset_ids(trecs, "scope", scope.get((model, retr))))
            g["true_verified"] = rate(tfl, tids, field="raw", value="VERIFIED")
            for name, ps in PATH_SETS.items():
                g[f"fcr_paths_{name}"] = rate(flags(trecs, paths=ps), tids)
            if "error" in sets:
                true_scores = [item_score(r) for r in trecs]
                g["auroc"] = round(auroc(err_scores, true_scores), 3)
                g["catch_at_fcr5"] = pct(catch_at_fcr(err_scores, true_scores, 0.05))
                rng = np.random.default_rng(0)
                boots = []
                e, t = np.array(err_scores), np.array(true_scores)
                for _ in range(1000):
                    boots.append(auroc(list(rng.choice(e, len(e))), list(rng.choice(t, len(t)))))
                g["auroc_ci"] = [round(float(np.nanpercentile(boots, 2.5)), 3),
                                 round(float(np.nanpercentile(boots, 97.5)), 3)]
        if "circular" in sets:
            crecs = sets["circular"][1]
            cfl = flags(crecs)
            cids = subset_ids(crecs, "all")
            g["circular_catch"] = rate(cfl, cids)
            for name, ps in PATH_SETS.items():
                g[f"circular_paths_{name}"] = rate(flags(crecs, paths=ps), cids)
            g["circular_path_of_catches"] = dict(Counter(cfl[i]["path"] for i in cids
                                                         if cfl[i]["gated"] == "CONTRADICTED"))
            if "error" in sets:
                efl = flags(sets["error"][1])
                pairs = [(c["pair"], c["id"]) for c in crecs if c.get("pair") in efl]
                b = sum(1 for e, c in pairs if efl[e]["gated"] == "CONTRADICTED" and cfl[c]["gated"] != "CONTRADICTED")
                cc = sum(1 for e, c in pairs if efl[e]["gated"] != "CONTRADICTED" and cfl[c]["gated"] == "CONTRADICTED")
                pe = np.mean([efl[e]["gated"] == "CONTRADICTED" for e, _ in pairs]) if pairs else float("nan")
                pc = np.mean([cfl[c]["gated"] == "CONTRADICTED" for _, c in pairs]) if pairs else float("nan")
                g["circular_inflation_matched"] = {
                    "n_pairs": len(pairs), "noncircular": pct(pe), "circular": pct(pc),
                    "delta_points": pct(pc - pe), "mcnemar_p": mcnemar_exact(b, cc)}
                g["circular_matched_by_paths"] = {}
                for name, ps in PATH_SETS.items():
                    ef = flags(sets["error"][1], paths=ps); cf = flags(crecs, paths=ps)
                    if not pairs:
                        continue
                    ce = [ef[e]["gated"] == "CONTRADICTED" for e, _ in pairs]
                    cc_ = [cf[c]["gated"] == "CONTRADICTED" for _, c in pairs]
                    bb = sum(1 for x, y in zip(ce, cc_) if x and not y)
                    cc2 = sum(1 for x, y in zip(ce, cc_) if y and not x)
                    g[f"circular_inflation_{name}"] = pct(np.mean(cc_) - np.mean(ce))
                    g["circular_matched_by_paths"][name] = {
                        "n_pairs": len(pairs), "circular": pct(np.mean(cc_)), "noncircular": pct(np.mean(ce)),
                        "delta_points": pct(np.mean(cc_) - np.mean(ce)), "mcnemar_p": mcnemar_exact(bb, cc2)}
                g["overlap"] = tokens_overlap_terciles(sets["error"][1], crecs, efl, cfl)
        summary["groups"][f"{arm}|{model}|{retr}"] = g

    # paired comparisons on identical items (exact McNemar)
    pairs = [(a, r) for a, r in (p.split(":") for p in args.pairs)]
    for (arm, model, retr), sets in sorted(groups.items()):
        scope.fam = family(arm)
        for want, ref_arm in pairs:
            if arm != want:
                continue
            ref = groups.get((ref_arm, model, retr))
            if not ref or "error" not in sets or "error" not in ref:
                continue
            a = flags(sets["error"][1]); r = flags(ref["error"][1])
            for sub in ("all", "heldout", "scope"):
                ids = [i for i in subset_ids(sets["error"][1], sub, scope.get((model, retr))) if i in r]
                ra = [a[i]["gated"] == "CONTRADICTED" for i in ids]
                rr = [r[i]["gated"] == "CONTRADICTED" for i in ids]
                b = sum(1 for x, y in zip(rr, ra) if x and not y)   # ref caught, arm missed
                c = sum(1 for x, y in zip(rr, ra) if y and not x)   # arm caught, ref missed
                both = sum(1 for x, y in zip(rr, ra) if x and y)
                n = len(ids)
                d = (c - b) / n if n else float("nan")
                se = math.sqrt(max(b + c - (c - b) ** 2 / n, 0)) / n if n else float("nan")
                summary["paired"][f"{arm} vs {ref_arm}|{model}|{retr}|{sub}"] = {
                    "n": n, "both": both, "fixed": c, "regressed": b, "neither": n - both - b - c,
                    "delta_points": pct(d), "delta_ci_wald": [pct(d - 1.96 * se), pct(d + 1.96 * se)],
                    "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    write_tables(summary, args.out)
    try:
        write_figures(runs, groups, args.out)
    except Exception as e:  # figures are a convenience; numbers are in summary.json
        print(f"(figures skipped: {e})")
    print(f"Saved {os.path.join(args.out, 'summary.json')} and tables.tex")
    for k, g in summary["groups"].items():
        c = g.get("catch_all", {})
        print(f"  {k:55s} catch {c.get('rate')}% {c.get('ci')}  FCR {g.get('fcr', {}).get('rate')}%  "
              f"AUROC {g.get('auroc')}  circular {g.get('circular_catch', {}).get('rate')}%  "
              f"replay-mismatch {g.get('replay_mismatches')}")


def fmt(r):
    if not r or r.get("rate") is None:
        return "--"
    return f"{r['rate']:.1f} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]"


def write_tables(summary, out):
    lines = ["% generated by exp/analyze.py -- do not edit by hand",
             "% Construction arms: catch (fixed gate), FCR, AUROC, catch@FCR5, circular",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "KB & Verifier & Catch & FCR & AUROC & C@5\\% & Circular \\\\", "\\midrule"]
    for k, g in summary["groups"].items():
        arm, model, retr = k.split("|")
        if retr != "concept":
            continue
        lines.append(f"{arm.replace('_', ' ')} & {model} & {fmt(g.get('catch_all'))} & {fmt(g.get('fcr'))} & "
                     f"{g.get('auroc', '--')} & {g.get('catch_at_fcr5', '--')} & {fmt(g.get('circular_catch'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "% Paired comparisons vs reference arm (exact McNemar)"]
    for k, p in summary["paired"].items():
        lines.append(f"% {k}: n={p['n']} fixed={p['fixed']} regressed={p['regressed']} "
                     f"delta={p['delta_points']} CI={p['delta_ci_wald']} p={p['mcnemar_exact_p']}")
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(lines) + "\n")


# Validated categorical palette (dataviz reference instance, light mode); color follows
# the KB arm everywhere, and every figure also carries a legend plus a second cue
# (marker / line style / hatch) because three slots sit below 3:1 on white.
ARM_STYLE = {
    "baseline_rule":  ("Rule-based",     "#2a78d6", "o", "-"),
    "kca":            ("KCA",            "#eb6834", "s", "-"),
    "sentence_split": ("Sentence split", "#1baf7a", "^", "--"),
    "llm_only":       ("LLM-only",       "#eda100", "D", "-."),
    "kca_regate_minicheck_flan_t5_large": ("KCA, re-gated", "#e87ba4", "v", ":"),
}
INK, MUTED = "#2b2b2b", "#8a8a85"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.yaxis.grid(True, color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)


def write_figures(runs, groups, out, main_model="roberta-large-mnli"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Elsevier artwork format: Arial-compatible font, >= 7 pt at final size, embedded fonts
    plt.rcParams.update({"font.size": 7.5, "font.family": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
    figdir = out
    sel = {k: v for k, v in groups.items() if k[1] == main_model and k[2] == "concept"}

    # 1. catch rate vs false-contradiction rate over the NLI threshold
    fig, ax = plt.subplots(figsize=(90 / 25.4, 70 / 25.4))      # 90 mm = one column
    for arm, (lab, col, mk, ls) in ARM_STYLE.items():
        sets = sel.get((arm, main_model, "concept"))
        if not sets or "error" not in sets or "true" not in sets:
            continue
        e = [item_score(r) for r in sets["error"][1]]
        t = [item_score(r) for r in sets["true"][1]]
        ths = np.linspace(0, 1.0, 101)
        xs = [np.mean([v >= x for v in t]) for x in ths]
        ys = [np.mean([v >= x for v in e]) for x in ths]
        ax.plot(xs, ys, color=col, ls=ls, lw=2, label=lab)
        i = int(round(DEFAULTS["nli_contra_threshold"] * 100))
        ax.plot([xs[i]], [ys[i]], marker=mk, ms=6, color=col, mec="white", mew=1)
    ax.plot([0, 1], [0, 1], ls=":", c=MUTED, lw=0.8)
    ax.set_xlabel("False-contradiction rate (true statements)")
    ax.set_ylabel("Catch rate (misconceptions)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _style(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_roc.pdf")); plt.close(fig)

    # 2. circular vs non-circular, by contradiction paths, for each KB arm that has both
    labels_p = [("all", "All"), ("nli_fact+nli_neg", "(a)+(b)"), ("nli_fact_only", "(a)")]
    arms = [a for a in ("baseline_rule", "kca") if (a, main_model, "concept") in sel
            and {"error", "circular"} <= set(sel[(a, main_model, "concept")])]
    if arms:
        fig, axes = plt.subplots(1, len(arms), figsize=(3.4, 2.3), sharey=True, squeeze=False)
        for ax, arm in zip(axes[0], arms):
            sets = sel[(arm, main_model, "concept")]
            ef_all = {n: flags(sets["error"][1], paths=PATH_SETS[n]) for n, _ in labels_p}
            cf_all = {n: flags(sets["circular"][1], paths=PATH_SETS[n]) for n, _ in labels_p}
            pairs = [(c["pair"], c["id"]) for c in sets["circular"][1]]
            pairs = [(e, c) for e, c in pairs if e in ef_all["all"]]
            x = np.arange(len(labels_p)); w = 0.36
            non = [np.mean([ef_all[n][e]["gated"] == "CONTRADICTED" for e, _ in pairs]) for n, _ in labels_p]
            cir = [np.mean([cf_all[n][c]["gated"] == "CONTRADICTED" for _, c in pairs]) for n, _ in labels_p]
            ax.bar(x - w / 2 - 0.01, non, w, color="#2a78d6", label="Independent misconceptions")
            ax.bar(x + w / 2 + 0.01, cir, w, color="#eb6834", hatch="///", edgecolor="white", lw=0,
                   label="Circular (stored negations)")
            for xi, v in zip(x, cir):
                ax.text(xi + w / 2, v + 0.02, f"{100*v:.0f}", ha="center", fontsize=6, color=INK)
            for xi, v in zip(x, non):
                ax.text(xi - w / 2, v + 0.02, f"{100*v:.0f}", ha="center", fontsize=6, color=INK)
            ax.set_xticks(x); ax.set_xticklabels([l for _, l in labels_p], fontsize=6)
            ax.set_title(ARM_STYLE[arm][0], fontsize=8, color=INK)
            ax.set_ylim(0, 1.12); _style(ax)
        axes[0][0].set_ylabel("Catch rate (matched concepts)")
        handles, labs = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=2, fontsize=6.5, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(os.path.join(figdir, "fig_circular_paths.pdf")); plt.close(fig)

    # 3. mean NLI contradiction probability by stored-fact length (on-topic pairs)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    bands = ["<=8 words", "9-15", "16-30", ">30"]
    drew = False
    for k, arm in enumerate(("baseline_rule", "kca")):
        sets = sel.get((arm, main_model, "concept"))
        if not sets or "error" not in sets:
            continue
        m = mechanism(sets["error"][1]).get("mean_contra_by_length")
        if not m:
            continue
        lab, col, mk, ls = ARM_STYLE[arm]
        ys = [m.get(b) for b in bands]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], color=col, ls=ls, lw=2, marker=mk, ms=6, mec="white", label=lab)
        drew = True
    if drew:
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(["\u22648", "9\u201315", "16\u201330", ">30"])
        ax.set_xlabel("Stored fact length (words)"); ax.set_ylabel("Mean $P_{con}$ (on-topic pairs)")
        _style(ax); ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_length.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
