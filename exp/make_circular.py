"""
make_circular.py -- the circular test set for ONE knowledge base.

Circularity is relative to the KB being evaluated: a circular item is that
KB's own stored negation. So each KB arm gets its own circular set, built for
the same 150 concepts as the non-circular set (concept-matched design).

Selection rule: first positive fact of the concept that carries a stored negation; else a
negation-flagged fact; else a rule negation of the first stored claim.
Only the concept's own facts are used.

USAGE:
  python3 exp/make_circular.py --kb exp/kbs/kca.jsonl --out exp/data/circular__kca.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from exp.kb_store import JsonKB  # noqa: E402
from core.extraction_engine import auto_generate_negation  # noqa: E402


def circular_text(facts):
    for f in facts:
        if not f.get("is_negation") and (f.get("negation") or "").strip():
            return f["negation"].strip(), f["claim"].strip(), f["fact_id"]
    for f in facts:
        if f.get("is_negation") and (f.get("claim") or "").strip():
            return f["claim"].strip(), f["claim"].strip(), f["fact_id"]
    for f in facts:
        if (f.get("claim") or "").strip():
            return auto_generate_negation(f["claim"].strip()), f["claim"].strip(), f["fact_id"]
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--concepts", default=os.path.join(ROOT, "exp", "data", "errors_noncircular.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kb = JsonKB(args.kb)
    base = json.load(open(args.concepts, encoding="utf-8"))
    items, missing = [], 0
    for e in base:
        text, src, fid = circular_text(kb.get_facts_for_concept(e["concept"]))
        if not text:
            missing += 1
            continue
        items.append({"id": "c" + e["id"][1:], "pair": e["id"], "concept": e["concept"], "text": text,
                      "kind": "circular", "source_claim": src, "source_fact_id": fid,
                      "rule_fallback_form": text.startswith("It is NOT the case that:"),
                      "in_dev55": e.get("in_dev55", False), "dup_cluster": e.get("dup_cluster")})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(items, open(args.out, "w"), indent=1, ensure_ascii=False)
    fb = sum(i["rule_fallback_form"] for i in items)
    print(f"{args.out}: {len(items)} circular items ({missing} concepts absent from this KB); "
          f"{fb} use the 'It is NOT the case that:' fallback form")


if __name__ == "__main__":
    main()
