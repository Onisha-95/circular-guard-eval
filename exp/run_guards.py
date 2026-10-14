"""
run_guards.py -- run a published, off-the-shelf grounding guard on the same KBs
and test sets as the main guard.

The main study varies the NLI model INSIDE one guard design (stored negations,
three contradiction paths, relevance gate). This script swaps the WHOLE guard
for one that other people built and use, so the circular-test effect can be
checked across guard designs:

  minicheck   MiniCheck-Flan-T5-Large (Tang, Laban & Durrett, EMNLP 2024)
  hhem        Vectara HHEM-2.1-Open (hallucination evaluation model)
  stub        word-overlap stand-in, for testing the pipeline without models

Both real guards answer one question: do the retrieved facts SUPPORT the
statement? A statement is FLAGGED when the support probability is below 0.5
(each model's own cut-off). Neither guard reads the stored negations.

Retrieval is the same as the main study (concept label, plus embedding
retrieval when the concept has fewer than 3 facts), and the retrieved facts are
given to the guard as one document, one fact per line.

USAGE
  python3 exp/run_guards.py --guard minicheck --kb exp/kbs/kca.jsonl \
      --items exp/data/errors_noncircular.json --out exp/runs_guards/kca__minicheck__errors.jsonl
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

THRESHOLD = 0.5


# ------------------------------------------------------------------ guards
def _no_device_map():
    """Load models on the default device (device_map='auto' fails on laptops short of memory)."""
    import transformers
    for cls in (transformers.AutoModelForSeq2SeqLM, transformers.AutoModelForSequenceClassification):
        orig = cls.from_pretrained
        if getattr(orig, "_patched", False):
            continue

        def fp(*a, _orig=orig, **k):
            k.pop("device_map", None)
            return _orig(*a, **k)
        fp._patched = True
        cls.from_pretrained = fp


def make_guard(name):
    """Return support(doc, statement) -> probability that the doc supports the statement."""
    if name == "minicheck":
        _no_device_map()
        try:
            import nltk
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from minicheck.minicheck import MiniCheck
        mc = MiniCheck(model_name="flan-t5-large",
                       cache_dir=os.path.join(ROOT, "exp", "cache", "minicheck"))

        def support(doc, statement):
            _, probs, _, _ = mc.score(docs=[doc], claims=[statement])
            return float(probs[0])
        return support, "MiniCheck-Flan-T5-Large"

    if name == "hhem":
        _no_device_map()
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True)
        model.eval()

        def support(doc, statement):
            out = model.predict([(doc, statement)])      # 1.0 = consistent with the premise
            return float(out[0])
        return support, "Vectara HHEM-2.1-Open"

    if name == "stub":                                   # tests only
        import re

        def toks(s):
            return set(re.findall(r"[a-z0-9]+", s.lower()))

        def support(doc, statement):
            t = toks(statement)
            if re.search(r"\bnot\b|n't", statement.lower()):
                return 0.1
            return len(t & toks(doc)) / max(len(t), 1)
        return support, "stub word overlap"
    raise SystemExit(f"unknown guard {name}")


# ---------------------------------------------------------------- retrieval
def retrieve(kb, rag, concept, fallback=True):
    """Same protocol as exp/run_eval.py --retrieval concept."""
    facts = list(kb.get_facts_for_concept(concept))
    used_fallback = False
    if len(facts) < 3 and fallback and rag is not None:
        used_fallback = True
        have = {f.get("fact_id") for f in facts}
        for f in rag.retrieve_context(f"Can you explain {concept} to me?", top_k=5):
            if f.get("fact_id") not in have:
                facts.append(f)
    return facts, used_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", required=True, choices=["minicheck", "hhem", "stub"])
    ap.add_argument("--kb", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fallback", action="store_true", help="tests only")
    ap.add_argument("--retrieval", choices=["concept", "open"], default="concept",
                    help="open: the facts most similar to the item text, no concept label")
    ap.add_argument("--open-k", type=int, default=10)
    args = ap.parse_args()

    from exp.kb_store import JsonKB, RagAdapter
    kb = JsonKB(args.kb)
    rag = None if args.no_fallback else RagAdapter(kb)
    support, guard_name = make_guard(args.guard)

    items = json.load(open(args.items, encoding="utf-8"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    with open(args.out + ".part", "w", encoding="utf-8") as out:
        out.write(json.dumps({"_meta": True, "guard": args.guard, "guard_name": guard_name,
                              "kb": args.kb, "items": args.items, "threshold": THRESHOLD,
                              "retrieval": args.retrieval}) + "\n")
        for n, it in enumerate(items, 1):
            if args.retrieval == "open":
                facts, fb = rag.retrieve_context(it["text"], top_k=args.open_k), False
            else:
                facts, fb = retrieve(kb, rag, it.get("concept", ""), fallback=not args.no_fallback)
            claims = [(f.get("claim") or "").strip() for f in facts if (f.get("claim") or "").strip()]
            rec = {k: it.get(k) for k in ("id", "pair", "kind", "concept", "text")}
            rec.update(n_facts=len(claims), fallback=fb)
            if claims:
                p = support("\n".join(claims), it["text"])
                rec.update(support=round(p, 5), flagged=p < THRESHOLD)
            else:
                rec.update(support=None, flagged=False)   # nothing to check against: passes
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(items):
                print(f"  {n}/{len(items)}  {(time.time() - t0) / n:.1f}s/item", flush=True)
    os.replace(args.out + ".part", args.out)
    print(f"Saved {args.out}  [{guard_name}]")


if __name__ == "__main__":
    main()
