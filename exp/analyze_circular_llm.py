"""
analyze_circular_llm.py -- circular gap with LLM-written circular errors.

For each KB and checker it compares, on the SAME concepts (matched pairs):
  independent misconceptions  vs  rule-made circular errors (main study)
  independent misconceptions  vs  LLM-written circular errors (make_circular_llm.py)
with exact McNemar tests, for KBGuard (all paths and path (a) only) and for
the published guards (MiniCheck, HHEM).

Items marked "n" in exp/data/circular_llm__<kb>__review.csv (not actually false)
are left out of every comparison, together with their pairs.

USAGE
  python3 exp/analyze_circular_llm.py            -> prints a table and writes
                                                    exp/results/circular_llm.json
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from exp.analyze import flags, mcnemar_exact, PATH_SETS  # noqa: E402

KBS = ["kca", "baseline_rule"]
MODELS = ["roberta-large-mnli", "llm:qwen/qwen3.8-flash@https://openrouter.ai/api/v1"]
GUARDS = ["minicheck", "hhem"]


def read(p):
    lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return lines[0], lines[1:]


def find_run(dirs, kb, model, kind):
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.jsonl"))):
            meta, recs = read(p)
            if not meta.get("_meta"):
                continue
            if os.path.basename(meta["kb"]).replace(".jsonl", "") != kb:
                continue
            if meta.get("nli_model") != model or meta.get("retrieval", "concept") != "concept":
                continue
            kinds = {r.get("kind") for r in recs}
            if kind in kinds:
                return recs
    return None


TAG = "circular_llm"


def rejected(kb):
    p = os.path.join(ROOT, "exp", "data", f"{TAG}__{kb}__review.csv")
    if not os.path.exists(p):
        return set(), False
    out, any_filled = set(), False
    for row in csv.DictReader(open(p, encoding="utf-8-sig")):
        v = (row.get("is_false_given_source (y/n)") or "").strip().lower()
        any_filled |= bool(v)
        if v.startswith("n"):
            out.add(row["id"])
    return out, any_filled


def tok(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jac(a, b):
    A, B = tok(a), tok(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def paired(e_flag, c_flag, pairs):
    ce = [e_flag[e] for e, _ in pairs]
    cc = [c_flag[c] for _, c in pairs]
    b = sum(1 for x, y in zip(ce, cc) if x and not y)
    c = sum(1 for x, y in zip(ce, cc) if y and not x)
    return {"n": len(pairs), "independent": round(100 * np.mean(ce), 1),
            "circular": round(100 * np.mean(cc), 1),
            "gap": round(100 * (np.mean(cc) - np.mean(ce)), 1), "p": mcnemar_exact(b, c)}


def main():
    global TAG
    if len(sys.argv) > 1:
        TAG = sys.argv[1]
    res = {}
    for kb in KBS:
        lp = os.path.join(ROOT, "exp", "data", f"{TAG}__{kb}.json")
        if not os.path.exists(lp):
            print(f"[{kb}] no LLM circular set; run make_circular_llm.py --kb {kb}")
            continue
        llm_items = json.load(open(lp, encoding="utf-8"))
        bad, reviewed = rejected(kb)
        keep_pairs = {it["pair"] for it in llm_items if it["id"] not in bad}
        rule_by_pair = {}
        rp = os.path.join(ROOT, "exp", "data", f"circular__{kb}.json")
        for it in json.load(open(rp, encoding="utf-8")):
            rule_by_pair[it["pair"]] = it
        res[kb] = {"n_llm_items": len(llm_items), "rejected_by_review": len(bad),
                   "review_filled": reviewed,
                   "overlap_with_source_fact": {
                       "rule": round(float(np.mean([jac(rule_by_pair[i["pair"]]["text"], i["source_claim"])
                                                    for i in llm_items if i["pair"] in rule_by_pair])), 2),
                       "llm": round(float(np.mean([jac(i["text"], i["source_claim"]) for i in llm_items])), 2)},
                   "rows": []}
        for m in MODELS:
            err = find_run(["exp/runs"], kb, m, "error")
            rule = find_run(["exp/runs"], kb, m, "circular")
            llm = find_run(["exp/runs_circllm"], kb, m, TAG)
            if not (err and llm):
                print(f"[{kb}] {m}: missing runs (errors={bool(err)}, circular_llm={bool(llm)})")
                continue
            for pname in ("all", "nli_fact_only"):
                ps = PATH_SETS[pname]
                ef = {k: v["gated"] == "CONTRADICTED" for k, v in flags(err, paths=ps).items()}
                lf = {k: v["gated"] == "CONTRADICTED" for k, v in flags(llm, paths=ps).items()}
                lpairs = [(i["pair"], i["id"]) for i in llm if i["pair"] in ef and i["pair"] in keep_pairs]
                row = {"checker": m.split("@")[0], "paths": pname, "llm_circular": paired(ef, lf, lpairs)}
                if rule:
                    rf = {k: v["gated"] == "CONTRADICTED" for k, v in flags(rule, paths=ps).items()}
                    rpairs = [(i["pair"], i["id"]) for i in rule if i["pair"] in ef and i["pair"] in keep_pairs]
                    row["rule_circular"] = paired(ef, rf, rpairs)
                res[kb]["rows"].append(row)
        for g in GUARDS:
            base = os.path.join(ROOT, "exp", "runs_guards")
            pe = os.path.join(base, f"{kb}__{g}__errors.jsonl")
            pr = os.path.join(base, f"{kb}__{g}__circular.jsonl")
            pl = os.path.join(ROOT, "exp", "runs_guards_circllm", f"{kb}__{g}__{TAG}.jsonl")
            if not (os.path.exists(pe) and os.path.exists(pl)):
                print(f"[{kb}] {g}: missing guard runs")
                continue
            ef = {r["id"]: bool(r["flagged"]) for r in read(pe)[1]}
            lrecs = read(pl)[1]
            lf = {r["id"]: bool(r["flagged"]) for r in lrecs}
            lpairs = [(r["pair"], r["id"]) for r in lrecs if r.get("pair") in ef and r["pair"] in keep_pairs]
            row = {"checker": g, "paths": "-", "llm_circular": paired(ef, lf, lpairs)}
            if os.path.exists(pr):
                rrecs = read(pr)[1]
                rf = {r["id"]: bool(r["flagged"]) for r in rrecs}
                rpairs = [(r["pair"], r["id"]) for r in rrecs if r.get("pair") in ef and r["pair"] in keep_pairs]
                row["rule_circular"] = paired(ef, rf, rpairs)
            res[kb]["rows"].append(row)

    os.makedirs(os.path.join(ROOT, "exp", "results"), exist_ok=True)
    out = os.path.join(ROOT, "exp", "results", f"{TAG}.json")
    json.dump(res, open(out, "w"), indent=1)
    for kb, r in res.items():
        print(f"\n=== {kb}: {r['n_llm_items']} LLM-written circular items, "
              f"{r['rejected_by_review']} rejected by hand review"
              + ("" if r["review_filled"] else "  (review CSV not filled yet)"))
        print(f"word overlap with the source fact: rule {r['overlap_with_source_fact']['rule']}, "
              f"LLM {r['overlap_with_source_fact']['llm']}")
        print(f"{'checker':28s} {'paths':14s} {'n':>4s} {'indep':>6s} {'rule':>6s} {'gap':>6s} {'p':>7s}"
              f" {'LLM':>6s} {'gap':>6s} {'p':>7s}")
        for row in r["rows"]:
            L, R = row["llm_circular"], row.get("rule_circular")
            rs = f"{R['circular']:6.1f} {R['gap']:+6.1f} {R['p']:7.3g}" if R else f"{'':6s} {'':6s} {'':7s}"
            print(f"{row['checker']:28s} {row['paths']:14s} {L['n']:4d} {L['independent']:6.1f} {rs}"
                  f" {L['circular']:6.1f} {L['gap']:+6.1f} {L['p']:7.3g}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
