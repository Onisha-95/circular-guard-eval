"""Cohen's kappa for the two-annotator review of the correct statements.

  python3 kappa_true_statements.py true_statements_review_annotatorA.xlsx true_statements_review_annotatorB.xlsx

Prints raw agreement and Cohen's kappa for both questions, lists every disagreement,
and writes exp/data/true_statements.json with a 'review' field on every item:
  kept       both annotators: Correct + Yes
  fix        at least one annotator: Incorrect / No / Unsure  (fix these, then re-run)
"""
import json, os, sys
from collections import Counter
from openpyxl import load_workbook

def read(path):
    ws = load_workbook(path, data_only=True)["Review"]
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0]:
            out[r[0]] = {"correct": (r[4] or "").strip(), "pair": (r[5] or "").strip(),
                         "fix": r[6] or "", "note": r[7] or ""}
    return out

def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0

A, B = read(sys.argv[1]), read(sys.argv[2])
ids = sorted(set(A) & set(B))
missing = [i for i in ids if not (A[i]["correct"] and A[i]["pair"] and B[i]["correct"] and B[i]["pair"])]
if missing:
    print(f"{len(missing)} rows are not filled in yet, e.g. {missing[:5]}"); sys.exit(1)
for q, name in (("correct", "Is the statement correct?"), ("pair", "Is it a fair pair?")):
    po, k = kappa([A[i][q] for i in ids], [B[i][q] for i in ids])
    print(f"{name:28s} agreement {100*po:.1f}%   Cohen's kappa {k:.2f}   (n={len(ids)})")
dis = [i for i in ids if (A[i]["correct"], A[i]["pair"]) != (B[i]["correct"], B[i]["pair"])]
print(f"\n{len(dis)} disagreements:")
for i in dis:
    print(f"  {i}: A={A[i]['correct']}/{A[i]['pair']}  B={B[i]['correct']}/{B[i]['pair']}  {A[i]['fix'] or B[i]['fix']}")

root = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(root, "exp", "data", "true_statements_DRAFT.json")
items = json.load(open(src))
for it in items:
    a, b = A.get(it["id"]), B.get(it["id"])
    ok = a and b and a["correct"] == b["correct"] == "Correct" and a["pair"] == b["pair"] == "Yes"
    it["review"] = "kept" if ok else "fix"
    it["status"] = "reviewed by two annotators"
out = os.path.join(root, "exp", "data", "true_statements.json")
json.dump(items, open(out, "w"), indent=1, ensure_ascii=False)
n_fix = sum(it["review"] == "fix" for it in items)
print(f"\nWrote {out}: {len(items) - n_fix} kept, {n_fix} to fix.")
if n_fix:
    print("Fix those statements in true_statements.json (both annotators agree on the new wording), set review to 'kept', then re-run the __true runs.")
