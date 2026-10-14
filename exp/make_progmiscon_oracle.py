"""
make_progmiscon_oracle.py -- a control that separates two causes of the circular gap.

The circular test differs from a real test in two ways at once:
  (1) coverage: the KB always holds the exact fact the error contradicts, and
  (2) wording:  the error is a template negation of that fact, not a mistake a
                person would make.
This script removes (1). It builds an "oracle" KB whose 85 facts are the
CORRECT statements from ProgMiscon, so every misconception has its exact
correction in the KB. It then builds two error sets on the same 85 facts:
  expert    ProgMiscon's own incorrect statement (written by experts from
            observed student mistakes)       -> exp/data/progmiscon_errors.json
  template  the KB's rule-based negation of the same correct statement
            (how circular items are made)     -> exp/data/progmiscon_oracle_template.json
Same facts, same KB, same retrieval, same guard: only the wording of the
error changes, so a paired test of expert vs template catch isolates (2).

  python3 exp/make_progmiscon_oracle.py
"""
import json
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
from core.extraction_engine import auto_generate_negation  # noqa: E402

D = os.path.join(ROOT, "exp", "data")


def main():
    errs = json.load(open(os.path.join(D, "progmiscon_errors.json"), encoding="utf-8"))
    trues = {t["id"]: t for t in json.load(open(os.path.join(D, "progmiscon_true.json"), encoding="utf-8"))}
    kb, tmpl = [], []
    for e in errs:
        t = trues[e["pair"]]
        claim = t["text"].strip()
        neg = auto_generate_negation(claim)
        fid = str(uuid.uuid5(uuid.NAMESPACE_URL, "progmiscon-oracle/" + t["source"]))
        kb.append({"fact_id": fid, "concept": e["concept"], "claim": claim, "negation": neg,
                   "is_negation": False, "priority": "MEDIUM", "source_file": t["source"],
                   "source_location": "", "arm": "progmiscon_oracle"})
        tmpl.append({"id": "pc" + e["id"][2:], "pair": e["id"], "concept": e["concept"], "text": neg,
                     "kind": "circular", "source": t["source"], "source_fact_id": fid,
                     "rule_fallback_form": neg.startswith("It is NOT the case that:")})
    os.makedirs(os.path.join(ROOT, "exp", "kbs"), exist_ok=True)
    with open(os.path.join(ROOT, "exp", "kbs", "progmiscon_oracle.jsonl"), "w", encoding="utf-8") as fh:
        for f in kb:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    json.dump(tmpl, open(os.path.join(D, "progmiscon_oracle_template.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    fb = sum(x["rule_fallback_form"] for x in tmpl)
    print(f"oracle KB: {len(kb)} facts -> exp/kbs/progmiscon_oracle.jsonl")
    print(f"template errors: {len(tmpl)} ({fb} in the 'It is NOT the case that:' form)"
          f" -> exp/data/progmiscon_oracle_template.json")
    for x in tmpl[:3]:
        print("  ", x["text"])


if __name__ == "__main__":
    main()
