"""
analyze_progmiscon.py -- paired and coverage analysis for the ProgMiscon test.

ProgMiscon gives each misconception as a PAIR: an incorrect statement and its
corrected version, written by experts, nearly word-for-word the same. So for
every guard we can ask, per pair:
  error only   the guard flags the misconception and passes the correct version (right)
  true only    the guard flags the correct version and passes the misconception (backwards)
  both / neither
and test "error only" vs "true only" with an exact McNemar test. We also report
the paired win rate: how often the misconception gets the higher alarm score
(0.5 = chance).

Coverage: the course KBs were not written for ProgMiscon, so some items have no
relevant fact. Each pair's coverage is the highest cosine similarity between the
misconception and any retrieved fact (from the RoBERTa run log of the same KB;
every guard retrieves the same 10 facts). Results are split into coverage
thirds, so a reader can see whether discrimination improves when the KB does
hold related facts.

  python3 exp/analyze_progmiscon.py
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp.analyze import auroc, flags, item_score, mcnemar_exact, wilson  # noqa: E402

NAMES = {"roberta-large-mnli": "KBGuard (RoBERTa)", "llm:qwen/qwen3.8-flash": "Qwen judge",
         "minicheck": "MiniCheck", "hhem": "HHEM",
         "llm:openai/gpt-6-sol": "GPT-6 Sol"}


def read(p):
    lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return lines[0], lines[1:]


def load(runs, guard_runs):
    """(kb, guard) -> {'error': {pair_key: (flag, score, rec)}, 'true': {...}}"""
    out = defaultdict(lambda: {"error": {}, "true": {}, "circular": {}})
    for p in glob.glob(os.path.join(runs, "*.jsonl")):
        meta, recs = read(p)
        kb = os.path.basename(meta["kb"]).replace(".jsonl", "")
        g = meta["nli_model"].split("@")[0]
        fl = flags(recs)
        for r in recs:
            key = r["id"] if r["kind"] == "error" else r["pair"]
            out[(kb, g)][r["kind"]][key] = (fl[r["id"]]["gated"] == "CONTRADICTED", item_score(r), r)
    for p in glob.glob(os.path.join(guard_runs, "*.jsonl")):
        meta, recs = read(p)
        kb = os.path.basename(meta["kb"]).replace(".jsonl", "")
        for r in recs:
            key = r["id"] if r["kind"] == "error" else r["pair"]
            s = 1.0 - r["support"] if r.get("support") is not None else 0.0
            out[(kb, meta["guard"])][r["kind"]][key] = (bool(r["flagged"]), s, r)
    return out


def coverage(groups, kb):
    """pair key -> max cosine between the misconception and its retrieved facts."""
    d = groups[(kb, "roberta-large-mnli")]["error"]
    return {k: max((f.get("gate_sim", 0.0) for f in rec["facts"].values()), default=0.0)
            for k, (_, _, rec) in d.items()}


def paired(d, keys):
    e, t = d["error"], d["true"]
    keys = [k for k in keys if k in e and k in t]
    eo = sum(1 for k in keys if e[k][0] and not t[k][0])
    to = sum(1 for k in keys if t[k][0] and not e[k][0])
    both = sum(1 for k in keys if e[k][0] and t[k][0])
    wins = np.mean([1.0 if e[k][1] > t[k][1] else 0.5 if e[k][1] == t[k][1] else 0.0 for k in keys])
    pos, neg = [e[k][1] for k in keys], [t[k][1] for k in keys]
    ce = sum(e[k][0] for k in keys); ct = sum(t[k][0] for k in keys)
    return {"n": len(keys),
            "catch": round(100 * ce / len(keys), 1), "fcr": round(100 * ct / len(keys), 1),
            "catch_ci": [round(100 * x, 1) for x in wilson(ce, len(keys))],
            "error_only": eo, "true_only": to, "both": both,
            "neither": len(keys) - eo - to - both,
            "mcnemar_p": mcnemar_exact(eo, to),
            "paired_win": round(float(wins), 3), "auroc": round(auroc(pos, neg), 3)}


def oracle(groups):
    """Expert misconceptions vs template negations, KB = the exact corrections."""
    res = {}
    tmpl_form = {}
    p = os.path.join(ROOT, "exp", "data", "progmiscon_oracle_template.json")
    if os.path.exists(p):
        tmpl_form = {t["pair"]: t["rule_fallback_form"] for t in json.load(open(p, encoding="utf-8"))}
    print("\n== oracle KB (exact correction of every misconception is in the KB)")
    print(f"   {'guard':16s} {'n':>3s} {'expert':>7s} {'template':>9s} {'gap':>6s}  T-only E-only  p      | template: rule form / fallback form | hit@10")
    for (kb, g), d in sorted(groups.items()):
        e, c = d["error"], d["circular"]
        keys = [k for k in e if k in c]
        if not keys:
            continue
        ce = sum(e[k][0] for k in keys); cc = sum(c[k][0] for k in keys)
        to = sum(1 for k in keys if c[k][0] and not e[k][0])
        eo = sum(1 for k in keys if e[k][0] and not c[k][0])
        n = len(keys)
        a = {"n": n, "expert": round(100 * ce / n, 1), "template": round(100 * cc / n, 1),
             "expert_ci": [round(100 * x, 1) for x in wilson(ce, n)],
             "template_ci": [round(100 * x, 1) for x in wilson(cc, n)],
             "gap_points": round(100 * (cc - ce) / n, 1), "template_only": to, "expert_only": eo,
             "mcnemar_p": mcnemar_exact(to, eo)}
        for form, val in (("rule_form", False), ("fallback_form", True)):
            ks = [k for k in keys if tmpl_form.get(k) is val]
            if ks:
                a[form] = {"n": len(ks), "expert": round(100 * sum(e[k][0] for k in ks) / len(ks), 1),
                           "template": round(100 * sum(c[k][0] for k in ks) / len(ks), 1)}
        recs = [e[k][2] for k in keys if "retrieved_ids" in e[k][2]]
        if recs:
            import uuid
            hit = [str(uuid.uuid5(uuid.NAMESPACE_URL, "progmiscon-oracle/" + r["source"])) in r["retrieved_ids"]
                   for r in recs]
            a["hit_at_10"] = round(100 * float(np.mean(hit)), 1)
        res[g] = a
        rf, ff = a.get("rule_form", {}), a.get("fallback_form", {})
        print(f"   {NAMES.get(g, g):16s} {n:3d} {a['expert']:7.1f} {a['template']:9.1f} {a['gap_points']:+6.1f}"
              f"  {to:6d} {eo:6d}  {a['mcnemar_p']:.2g}  | {rf.get('expert')}->{rf.get('template')} (n={rf.get('n')})"
              f" / {ff.get('expert')}->{ff.get('template')} (n={ff.get('n')}) | {a.get('hit_at_10')}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "exp", "runs_progmiscon"))
    ap.add_argument("--guard-runs", default=os.path.join(ROOT, "exp", "runs_guards_progmiscon"))
    ap.add_argument("--oracle-runs", default=os.path.join(ROOT, "exp", "runs_progmiscon_oracle"))
    ap.add_argument("--oracle-guard-runs", default=os.path.join(ROOT, "exp", "runs_guards_progmiscon_oracle"))
    ap.add_argument("--raw", default=os.path.join(ROOT, "exp", "data", "progmiscon_raw.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "results", "progmiscon", "paired.json"))
    args = ap.parse_args()
    groups = load(args.runs, args.guard_runs)
    try:
        raw = json.load(open(args.raw, encoding="utf-8"))
        errs = json.load(open(os.path.join(os.path.dirname(args.raw), "progmiscon_errors.json"), encoding="utf-8"))
        observed = {it["id"] for it, r in zip(errs, raw) if r.get("observed")}
    except FileNotFoundError:      # ProgMiscon text not downloaded: run exp/fetch_progmiscon.py
        print("ProgMiscon data not found; the 'observed' subset is skipped")
        observed = set()

    res = {}
    for kb in sorted({k for k, _ in groups if k != "progmiscon_oracle"}):
        cov = coverage(groups, kb)
        keys = sorted(cov)
        cuts = np.quantile([cov[k] for k in keys], [1 / 3, 2 / 3])
        thirds = {"low": [k for k in keys if cov[k] < cuts[0]],
                  "mid": [k for k in keys if cuts[0] <= cov[k] < cuts[1]],
                  "high": [k for k in keys if cov[k] >= cuts[1]]}
        res[kb] = {"coverage": {"median": round(float(np.median(list(cov.values()))), 3),
                                "cuts": [round(float(c), 3) for c in cuts]}}
        print(f"\n== {kb}: coverage median {res[kb]['coverage']['median']}, thirds cut at {res[kb]['coverage']['cuts']}")
        print(f"   {'guard':16s} {'n':>3s} {'catch':>6s} {'FCR':>6s} {'AUROC':>6s} {'win':>6s}  E-only T-only both neither  p      | AUROC low/mid/high  | observed AUROC")
        for (k2, g), d in sorted(groups.items()):
            if k2 != kb:
                continue
            a = paired(d, keys)
            a["thirds"] = {t: paired(d, ks) for t, ks in thirds.items()}
            a["observed"] = paired(d, [k for k in keys if k in observed]) if observed else {"auroc": float("nan"), "n": 0}
            res[kb][g] = a
            th = " / ".join(f"{a['thirds'][t]['auroc']:.2f}" for t in ("low", "mid", "high"))
            print(f"   {NAMES.get(g, g):16s} {a['n']:3d} {a['catch']:6.1f} {a['fcr']:6.1f} {a['auroc']:6.3f} {a['paired_win']:6.3f}"
                  f"  {a['error_only']:6d} {a['true_only']:6d} {a['both']:4d} {a['neither']:7d}  {a['mcnemar_p']:.2g}"
                  f"  | {th}  | {a['observed']['auroc']:.3f} (n={a['observed']['n']})")
    og = load(args.oracle_runs, args.oracle_guard_runs)
    if og:
        res["oracle"] = oracle(og)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
