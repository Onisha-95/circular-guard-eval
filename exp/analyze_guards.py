"""
analyze_guards.py -- results for the off-the-shelf guards (exp/run_guards.py).

For every (KB, guard) it reports, with the same statistics as exp/analyze.py:
  catch rate on misconceptions, FCR on correct statements (Wilson 95% CIs),
  AUROC and catch@FCR<=5% using the score 1 - support,
  circular catch rate, and the concept-matched circular gap with exact McNemar p.

  python3 exp/analyze_guards.py            # reads exp/runs_guards/, writes exp/results/guards.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from exp.analyze import wilson, mcnemar_exact, auroc, catch_at_fcr  # noqa: E402

SETS = {"errors": "errors", "true": "true", "circular": "circular"}


def load(run_dir):
    groups = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        lines = [json.loads(l) for l in open(p, encoding="utf-8")]
        meta, recs = lines[0], lines[1:]
        kb = os.path.basename(meta["kb"]).replace(".jsonl", "")
        kind = os.path.basename(p).rsplit("__", 1)[-1].replace(".jsonl", "")
        groups[(kb, meta["guard"])][kind] = recs
        groups[(kb, meta["guard"])]["_name"] = meta.get("guard_name", meta["guard"])
    return groups


def rate(recs):
    k, n = sum(r["flagged"] for r in recs), len(recs)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(100 * k / n, 1) if n else None,
            "ci": [round(100 * lo, 1), round(100 * hi, 1)] if n else None}


def score(r):
    return 1.0 - r["support"] if r.get("support") is not None else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "exp", "runs_guards"))
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "results", "guards.json"))
    ap.add_argument("--accepted-only", action="store_true",
                    help="course true statements: only those both annotators accepted")
    args = ap.parse_args()
    groups = load(args.runs)
    if args.accepted_only:
        from exp.analyze import accepted_true_ids
        keep = accepted_true_ids()
        for (kb, _), d in groups.items():
            if "true" in d and not kb.startswith("fever_"):
                d["true"] = [r for r in d["true"] if r["id"] in keep]
    res = {}
    for (kb, guard), d in sorted(groups.items()):
        g = {"guard": d["_name"]}
        err, tru, cir = d.get("errors"), d.get("true"), d.get("circular")
        if err:
            g["catch"] = rate(err)
        if tru:
            g["fcr"] = rate(tru)
        if err and tru:
            pos, neg = [score(r) for r in err], [score(r) for r in tru]
            g["auroc"] = round(auroc(pos, neg), 3)
            g["catch_at_fcr5"] = round(100 * catch_at_fcr(pos, neg), 1)
        if cir:
            g["circular"] = rate(cir)
        if err and cir:
            e = {r["id"]: r["flagged"] for r in err}
            pairs = [(e[c["pair"]], c["flagged"]) for c in cir if c.get("pair") in e]
            b = sum(1 for x, y in pairs if not x and y)      # caught only when circular
            c = sum(1 for x, y in pairs if x and not y)      # caught only when independent
            n = len(pairs)
            g["circular_matched"] = {
                "n_pairs": n,
                "independent": round(100 * sum(x for x, _ in pairs) / n, 1),
                "circular": round(100 * sum(y for _, y in pairs) / n, 1),
                "gap_points": round(100 * (sum(y for _, y in pairs) - sum(x for x, _ in pairs)) / n, 1),
                "mcnemar_p": mcnemar_exact(b, c)}
        res[f"{kb}|{guard}"] = g
        cm = g.get("circular_matched", {})
        print(f"{kb:16s} {guard:10s} catch {g.get('catch', {}).get('rate')}  FCR {g.get('fcr', {}).get('rate')}  "
              f"AUROC {g.get('auroc')}  C@5 {g.get('catch_at_fcr5')}  circular {g.get('circular', {}).get('rate')}  "
              f"gap {cm.get('gap_points')} (p={cm.get('mcnemar_p', float('nan')):.2g})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
