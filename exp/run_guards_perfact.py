"""
run_guards_perfact.py -- the published guards (MiniCheck, HHEM) used fact by fact.

In the main study each published guard receives the retrieved facts joined into
ONE document and flags a statement when that document does not support it.
A reviewer may say that this is unfair: short slide facts joined together are not
the documents these guards were trained on. This script gives each guard the
setting most favourable to it: every retrieved fact is scored SEPARATELY, and a
statement counts as supported if ANY single fact supports it (support >= 0.5).

Retrieval is identical to exp/run_guards.py (concept label; embedding retrieval
when the concept has fewer than 3 facts). The script reports catch rate, FCR on
the accepted correct statements, AUROC, catch at FCR <= 5%, and the matched
circular gaps (rule-made and, if present, LLM-written), with exact McNemar p.

USAGE
  python3 exp/run_guards_perfact.py --guard minicheck --kb kca
  python3 exp/run_guards_perfact.py --guard hhem --kb baseline_rule
  python3 exp/run_guards_perfact.py --guard stub --kb kca        (pipeline test)
Outputs: exp/runs_guards_perfact/<kb>__<guard>__<set>.jsonl,
         exp/results/guards_perfact__<kb>__<guard>.json
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from exp.run_guards import make_guard, retrieve, THRESHOLD  # noqa: E402
from exp.run_chunk_verifier import rate, gap, auroc, catch_at  # noqa: E402


def reviewed_out(tag, kb):
    """ids marked n (not clearly false) in the hand-check file of an LLM-written set"""
    import csv
    p = os.path.join(ROOT, "exp", "data", f"{tag}__{kb}__review.csv")
    if not os.path.exists(p):
        return set()
    return {r["id"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))
            if (r.get("is_false_given_source (y/n)") or "").strip().lower().startswith("n")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", required=True, choices=["minicheck", "hhem", "stub"])
    ap.add_argument("--kb", required=True)
    ap.add_argument("--sets", default="all",
                    help="comma list of sets to run, e.g. errors,true (default: all available)")
    args = ap.parse_args()

    from exp.kb_store import JsonKB, RagAdapter
    kb = JsonKB(os.path.join(ROOT, "exp", "kbs", f"{args.kb}.jsonl"))
    rag = RagAdapter(kb)
    support, guard_name = make_guard(args.guard)

    data = os.path.join(ROOT, "exp", "data")
    keep = {it["id"] for it in json.load(open(os.path.join(data, "true_statements.json"), encoding="utf-8"))
            if it.get("review") == "kept"}
    sets = {"errors": json.load(open(os.path.join(data, "errors_noncircular.json"), encoding="utf-8")),
            "true": [it for it in json.load(open(os.path.join(data, "true_statements.json"), encoding="utf-8"))
                     if it["id"] in keep],
            "circular": json.load(open(os.path.join(data, f"circular__{args.kb}.json"), encoding="utf-8"))}
    for tag in ("circular_llm", "circular_llm_g2"):
        p = os.path.join(data, f"{tag}__{args.kb}.json")
        if os.path.exists(p):
            sets[tag] = json.load(open(p, encoding="utf-8"))

    if args.sets != "all":
        want = set(args.sets.split(","))
        sets = {k: v for k, v in sets.items() if k in want}
    outdir = os.path.join(ROOT, "exp", "runs_guards_perfact")
    os.makedirs(outdir, exist_ok=True)
    runs = {}
    for name, items in sets.items():
        path = os.path.join(outdir, f"{args.kb}__{args.guard}__{name}.jsonl")
        if os.path.exists(path):
            runs[name] = [json.loads(l) for l in open(path, encoding="utf-8")][1:]
            print(f"  reuse {path}")
            continue
        t0, recs = time.time(), []
        with open(path + ".part", "w", encoding="utf-8") as out:
            out.write(json.dumps({"_meta": True, "guard": args.guard, "guard_name": guard_name,
                                  "kb": args.kb, "mode": "per-fact max", "threshold": THRESHOLD}) + "\n")
            for n, it in enumerate(items, 1):
                facts, fb = retrieve(kb, rag, it.get("concept", ""))
                claims = [(f.get("claim") or "").strip() for f in facts if (f.get("claim") or "").strip()]
                rec = {k: it.get(k) for k in ("id", "pair", "kind", "concept", "text")}
                if claims:
                    ps = [support(c, it["text"]) for c in claims]
                    best = max(ps)
                    rec.update(n_facts=len(claims), fallback=fb, per_fact=[round(p, 4) for p in ps],
                               support=round(best, 5), flagged=best < THRESHOLD)
                else:
                    rec.update(n_facts=0, fallback=fb, support=None, flagged=False)
                recs.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 25 == 0 or n == len(items):
                    print(f"  {name}: {n}/{len(items)}  {(time.time() - t0) / n:.2f}s/item", flush=True)
        os.replace(path + ".part", path)
        runs[name] = recs

    err, tru = runs["errors"], runs["true"]
    sc = lambda r: 1 - r["support"] if r.get("support") is not None else 0.0  # noqa: E731
    pos, neg = [sc(r) for r in err], [sc(r) for r in tru]
    res = {"kb": args.kb, "guard": guard_name, "mode": "per-fact max",
           "catch": rate(err), "fcr": rate(tru), "auroc": auroc(pos, neg), "catch_at_fcr5": catch_at(pos, neg)}
    for name in runs:
        if name.startswith("circular"):
            bad = reviewed_out(name, args.kb) if name.startswith("circular_llm") else set()
            res[f"gap_{name}"] = gap(err, [r for r in runs[name] if r["id"] not in bad])
    os.makedirs(os.path.join(ROOT, "exp", "results"), exist_ok=True)
    rp = os.path.join(ROOT, "exp", "results", f"guards_perfact__{args.kb}__{args.guard}.json")
    json.dump(res, open(rp, "w"), indent=1)

    print(f"\n=== {guard_name}, per-fact, {args.kb} KB ===")
    print(f"catch {res['catch']['rate']}% {res['catch']['ci']}   FCR {res['fcr']['rate']}% {res['fcr']['ci']}   "
          f"AUROC {res['auroc']}   catch@FCR<=5% {res['catch_at_fcr5']}%")
    for k, g in res.items():
        if k.startswith("gap_") and g:
            print(f"{k:22s} n={g['n']}  independent {g['independent']}%  circular {g['circular']}%  "
                  f"gap {g['gap']:+.1f}  p={g['p']:.3g}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
