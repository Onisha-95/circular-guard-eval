"""
verify_paper_numbers.py -- checks every number in the paper that comes from the analyses run
on this computer after Oct 14 (LLM-written circular errors, second-annotator agreement,
per-fact scoring of the published guards, chunk-level verifier). It only READS files in
exp/results, exp/data and the run folders; it changes nothing.

USAGE (from the project folder, with .venv active):
    python3 exp/verify_paper_numbers.py
Every line ends in PASS or FAIL. Send the whole output back.
"""
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
RES = os.path.join(ROOT, "exp", "results")
fails = 0


def check(label, got, want, tol=0.05):
    global fails
    ok = got is not None and (abs(float(got) - float(want)) <= tol if isinstance(want, (int, float)) else got == want)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} paper={want!s:8s} file={got!s}")


def load(*names):
    for n in names:
        p = os.path.join(RES, n)
        if os.path.exists(p):
            return json.load(open(p)), n
    print(f"MISSING  none of {names}")
    return None, None


def p_cat(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ---------------------------------------------------------------- Table 5 (LLM-written circular errors)
want = {  # tag: kb: checker/paths: (gap, p as printed in the paper)
    "circular_llm": {
        "kca": {("roberta-large-mnli", "all"): (25.0, "<0.001"), ("roberta-large-mnli", "nli_fact_only"): (31.9, "<0.001"),
                ("llm:qwen/qwen3.8-flash", "all"): (5.6, "0.48"), ("minicheck", "-"): (-5.6, "0.22"), ("hhem", "-"): (-47.2, "<0.001")},
        "baseline_rule": {("roberta-large-mnli", "all"): (19.2, "0.013"), ("roberta-large-mnli", "nli_fact_only"): (28.8, "<0.001"),
                          ("llm:qwen/qwen3.8-flash", "all"): (13.7, "0.031"), ("minicheck", "-"): (-9.6, "0.065"), ("hhem", "-"): (-52.1, "<0.001")}},
    "circular_llm_g2": {
        "kca": {("roberta-large-mnli", "all"): (25.6, "<0.001"), ("roberta-large-mnli", "nli_fact_only"): (31.1, "<0.001"),
                ("llm:qwen/qwen3.8-flash", "all"): (-1.1, "1.00"), ("minicheck", "-"): (-3.3, "0.51"), ("hhem", "-"): (-43.3, "<0.001")},
        "baseline_rule": {("roberta-large-mnli", "all"): (24.0, "<0.001"), ("roberta-large-mnli", "nli_fact_only"): (30.0, "<0.001"),
                          ("llm:qwen/qwen3.8-flash", "all"): (10.0, "0.064"), ("minicheck", "-"): (-2.0, "0.75"), ("hhem", "-"): (-52.0, "<0.001")}},
}
n_checked = {"circular_llm": {"kca": 72, "baseline_rule": 73}, "circular_llm_g2": {"kca": 90, "baseline_rule": 100}}
valid_share = {"circular_llm": {"kca": 65, "baseline_rule": 61}, "circular_llm_g2": {"kca": 80, "baseline_rule": 84}}
overlaps = []
print("== Table 5: LLM-written circular errors")
for tag in want:
    d, _ = load(f"{tag}.json")
    if not d:
        fails += 1
        continue
    for kb in ("kca", "baseline_rule"):
        r = d.get(kb, {})
        n_items, rej = r.get("n_llm_items"), r.get("rejected_by_review")
        if n_items:
            check(f"{tag} {kb}: share clearly false (%)", round(100 * (n_items - rej) / n_items), valid_share[tag][kb], tol=0.5)
        overlaps.append(r.get("overlap_with_source_fact", {}).get("llm"))
        rows = {(row["checker"].split("@")[0], row["paths"]): row for row in r.get("rows", [])}
        rob = rows.get(("roberta-large-mnli", "all"))
        check(f"{tag} {kb}: checked items (n)", rob["llm_circular"]["n"] if rob else None, n_checked[tag][kb], tol=0)
        for key, (g, p) in want[tag][kb].items():
            row = rows.get(key)
            L = row["llm_circular"] if row else None
            check(f"{tag} {kb} {key[0].split('/')[-1]} {key[1]}: gap", L["gap"] if L else None, g)
            check(f"{tag} {kb} {key[0].split('/')[-1]} {key[1]}: p", p_cat(L["p"]) if L else None, p)
ov = [o for o in overlaps if o is not None]
if ov:
    check("word overlap of LLM edits, min", min(ov), 0.74, tol=0.005)
    check("word overlap of LLM edits, max", max(ov), 0.76, tol=0.005)

# unchecked gpt-4o-mini set: Qwen gap before the hand check (paper: -15.3)
print("== Unchecked gpt-4o-mini set (paper: Qwen gap -15.3 on one KB)")
try:
    import exp.analyze_circular_llm as A
    from exp.analyze import flags, PATH_SETS
    for kb in ("kca", "baseline_rule"):
        m = "llm:qwen/qwen3.8-flash@https://openrouter.ai/api/v1"
        err = A.find_run(["exp/runs"], kb, m, "error")
        llm = A.find_run(["exp/runs_circllm"], kb, m, "circular_llm")
        ef = {k: v["gated"] == "CONTRADICTED" for k, v in flags(err, paths=PATH_SETS["all"]).items()}
        lf = {k: v["gated"] == "CONTRADICTED" for k, v in flags(llm, paths=PATH_SETS["all"]).items()}
        pairs = [(i["pair"], i["id"]) for i in llm if i["pair"] in ef]
        print(f"INFO  {kb}: Qwen gap on ALL {len(pairs)} unchecked edits = {A.paired(ef, lf, pairs)['gap']:+.1f}")
except Exception as e:  # noqa: BLE001
    print("INFO  could not recompute the unchecked gap:", e)

# ---------------------------------------------------------------- second annotator
print("== Second annotator (TA)")
d, _ = load("circular_llm_agreement.json")
if d:
    check("agreement n", d["n"], 60, tol=0)
    check("agreement %", d["agreement"], 88.3)
    check("Cohen's kappa", round(d["cohen_kappa"], 2), 0.74, tol=0.005)
    check("Gwet's AC1", round(d["gwet_ac1"], 2), 0.79, tol=0.005)
    dis = d.get("disagreements", [])
    check("number of disagreements", len(dis), 7, tol=0)
    check("disagreements where TA was stricter (TA=n, first=y)", sum(x["first"] == "y" and x["second"] == "n" for x in dis), 7, tol=0)

# ---------------------------------------------------------------- per-fact scoring
print("== Published guards scored fact by fact")
pf = {}
for kb in ("kca", "baseline_rule"):
    for g in ("minicheck", "hhem"):
        d, _ = load(f"guards_perfact__{kb}__{g}.json")
        if d:
            pf[(kb, g)] = d
            print(f"INFO  {kb} {g}: catch {d['catch']['rate']}  FCR {d['fcr']['rate']}  AUROC {d['auroc']}")
for g, lo, hi, catch, auc in (("minicheck", 52.9, 61.8, 90.7, 0.80), ("hhem", 82.4, 84.6, 96.0, 0.70)):
    vals = [pf[(kb, g)] for kb in ("kca", "baseline_rule") if (kb, g) in pf]
    if vals:
        f = [v["fcr"]["rate"] for v in vals]
        check(f"{g} per-fact FCR min", min(f), lo)
        check(f"{g} per-fact FCR max", max(f), hi)
        for v in vals:
            check(f"{g} per-fact catch ({v['kb']})", v["catch"]["rate"], catch)
            check(f"{g} per-fact AUROC ({v['kb']}, 2 decimals)", round(v["auroc"], 2), auc, tol=0.005)

# ---------------------------------------------------------------- chunk-level verifier (Appendix B)
print("== Chunk-level verifier (Table B.1)")
tb = {("kca", "chunk"): (64.0, 18.4, 0.831, 36.7, -33.0, "<0.001", -5.6, "0.57"),
      ("kca", "claim"): (35.3, 16.9, 0.620, 8.0, 2.7, "0.76", 1.4, "1.00"),
      ("baseline_rule", "chunk"): (73.3, 22.8, 0.846, 44.0, -21.0, "<0.001", -19.2, "0.016"),
      ("baseline_rule", "claim"): (38.7, 14.7, 0.690, 20.7, 0.0, "1.00", -9.6, "0.31")}
for (kb, order), w in tb.items():
    suf = "" if order == "chunk" else "_claimfirst"
    d, name = load(f"eduguard_style__{kb}{suf}.json", f"chunk_verifier__{kb}{suf}.json")
    if not d:
        fails += 1
        continue
    lab = f"{kb} {order}-premise"
    check(f"{lab}: catch", d["catch"]["rate"], w[0])
    check(f"{lab}: FCR", d["fcr"]["rate"], w[1])
    check(f"{lab}: AUROC", round(d["auroc"], 3), w[2], tol=0.0005)
    check(f"{lab}: C@5", d["catch_at_fcr5"], w[3])
    cr, cl = d.get("circular_rule") or {}, d.get("circular_llm") or {}
    check(f"{lab}: rule-made gap", cr.get("gap"), w[4])
    check(f"{lab}: rule-made p", p_cat(cr["p"]) if cr else None, w[5])
    check(f"{lab}: LLM-written gap", cl.get("gap"), w[6])
    check(f"{lab}: LLM-written p", p_cat(cl["p"]) if cl else None, w[7])

print(f"\n{fails} FAIL(s)" if fails else "\nAll checks PASS")
