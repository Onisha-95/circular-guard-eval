"""
review_agreement.py -- independent second check of the LLM-written circular errors.

Step 1 (make a blind sheet for a second annotator):
  python3 exp/review_agreement.py sample --n 60
  -> exp/data/circular_llm_second_annotator.csv : a random sample (fixed seed) of
     LLM-written edits from every review file, shuffled, WITHOUT any existing label
     or helper column. The second annotator fills the last column with y/n.

Step 2 (agreement with the labels used in the paper):
  python3 exp/review_agreement.py agree
  -> percent agreement, Cohen's kappa, and Gwet's AC1 (robust to skewed labels),
     written to exp/results/circular_llm_agreement.json
"""
import argparse
import csv
import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "exp", "data")
COL = "is_false_given_source (y/n)"
SHEET = os.path.join(DATA, "circular_llm_second_annotator.csv")


def review_files():
    return sorted(glob.glob(os.path.join(DATA, "circular_llm*__review.csv")))


def first_labels():
    lab = {}
    for p in review_files():
        tag_kb = os.path.basename(p).replace("__review.csv", "")
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            v = (r.get(COL) or "").strip().lower()[:1]
            if v in ("y", "n"):
                lab[f"{tag_kb}:{r['id']}"] = (v, r)
    return lab


def sample(n):
    lab = first_labels()
    keys = sorted(lab)
    random.Random(2026).shuffle(keys)
    pick = keys[:n]
    with open(SHEET, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "source_fact", "llm_error", COL])
        for i, k in enumerate(pick, 1):
            r = lab[k][1]
            w.writerow([f"{i:03d}|{k}", " ".join(r["source_fact"].split()), r["llm_error"], ""])
    print(f"wrote {SHEET} ({len(pick)} items from {len(review_files())} review files). "
          f"Give it to the second annotator; they should not see the other files.")


def kappa_ac1(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(x == "y" for x in a) / n, sum(x == "y" for x in b) / n
    pe_k = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe_k) / (1 - pe_k) if pe_k < 1 else float("nan")
    pi = (pa + pb) / 2
    pe_g = 2 * pi * (1 - pi)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else float("nan")
    return po, kappa, ac1


def agree():
    lab = first_labels()
    a, b, rows = [], [], []
    for r in csv.DictReader(open(SHEET, encoding="utf-8-sig")):
        v = (r.get(COL) or "").strip().lower()[:1]
        if v not in ("y", "n"):
            continue
        key = r["item"].split("|", 1)[1]
        if key in lab:
            a.append(lab[key][0]); b.append(v)
            rows.append({"item": key, "first": lab[key][0], "second": v})
    if not a:
        raise SystemExit("no labelled rows in the second-annotator sheet yet")
    po, k, ac1 = kappa_ac1(a, b)
    res = {"n": len(a), "agreement": round(100 * po, 1), "cohen_kappa": round(k, 3), "gwet_ac1": round(ac1, 3),
           "first_yes": sum(x == "y" for x in a), "second_yes": sum(x == "y" for x in b),
           "disagreements": [r for r in rows if r["first"] != r["second"]]}
    os.makedirs(os.path.join(ROOT, "exp", "results"), exist_ok=True)
    out = os.path.join(ROOT, "exp", "results", "circular_llm_agreement.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"n={res['n']}  agreement {res['agreement']}%  Cohen's kappa {res['cohen_kappa']}  "
          f"Gwet's AC1 {res['gwet_ac1']}  ({len(res['disagreements'])} disagreements)\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "agree"])
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    sample(args.n) if args.step == "sample" else agree()
