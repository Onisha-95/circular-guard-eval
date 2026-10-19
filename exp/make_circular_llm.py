"""
make_circular_llm.py -- circular test errors written by an LLM instead of a rule.

The main circular set uses the KB's own rule-made negations ("is" -> "is NOT",
or "It is NOT the case that: ..."). A reviewer can fairly say that nobody tests
a guard with such templates. Published hallucination benchmarks more often ask
an LLM to make a small edit to a correct source statement so that it becomes
false (e.g. the perturbations used to build HaluBench for Lynx, Ravi et al., 2024).

This script applies that recipe to the SAME source facts as the rule-made
circular set: for every item in exp/data/circular__<kb>.json it asks an LLM to
make a minimal edit of `source_claim` that makes it false, WITHOUT simply adding
a negation. The result is still circular (made from the KB's own fact), but it
no longer uses the KB's stored negation or a template.

Output: exp/data/circular_llm__<kb>.json (same ids/pairs/concepts, kind
"circular_llm") and exp/data/circular_llm__<kb>__review.csv for a hand check.
Every LLM answer is cached in exp/cache/, so reruns cost nothing.

USAGE
  export OPENROUTER_API_KEY=...
  python3 exp/make_circular_llm.py --kb kca
  python3 exp/make_circular_llm.py --kb baseline_rule
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OR = "https://openrouter.ai/api/v1"

PROMPT = """Below is a correct statement taken from course notes for an introductory computing course.

Statement: {fact}

Rewrite it with one small change so that it becomes factually WRONG, the way a
student or a careless tutor might get it wrong. For example, change a term, a
number, a relation, or a direction, or swap two things. Rules:
- Do NOT just add "not", "no", "never" or "does not"; change the content instead.
- Keep the wording, length and style close to the original.
- The result must be clearly false according to the original statement.
Reply with the rewritten statement only, on one line."""

NEG = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|doesn't|don't|won't|didn't|wasn't|weren't)\b", re.I)


def make_llm(model):
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    client = OpenAI(api_key=key, base_url=OR)
    os.makedirs(os.path.join(ROOT, "exp", "cache"), exist_ok=True)
    cpath = os.path.join(ROOT, "exp", "cache",
                         "circular_llm__" + re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".jsonl")
    cache = {}
    if os.path.exists(cpath):
        for line in open(cpath, encoding="utf-8"):
            d = json.loads(line)
            cache[d["k"]] = d["v"]

    def call(prompt):
        k = hashlib.sha1((model + "\x00" + prompt).encode()).hexdigest()
        if k in cache:
            return cache[k]
        from exp.nli_backends import with_retries
        r = with_retries(lambda: client.chat.completions.create(
            model=model, temperature=0, max_tokens=120,
            messages=[{"role": "user", "content": prompt}]))
        v = (r.choices[0].message.content or "").strip()
        cache[k] = v
        with open(cpath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "v": v}) + "\n")
        return v
    return call


def clean(s):
    s = s.strip().splitlines()[0].strip() if s.strip() else ""
    s = re.sub(r'^(statement|rewritten statement)\s*:\s*', "", s, flags=re.I)
    return s.strip().strip('"').strip("“”").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="KB arm name, e.g. kca or baseline_rule")
    ap.add_argument("--model", default="openai/gpt-4o-mini",
                    help="generator (default: the LLM that built the KCA KB)")
    ap.add_argument("--tag", default="circular_llm",
                    help="output name and item kind; use a new tag for a second generator")
    ap.add_argument("--blind", action="store_true",
                    help="review CSV without any helper column (for an independent annotator)")
    args = ap.parse_args()

    src = os.path.join(ROOT, "exp", "data", f"circular__{args.kb}.json")
    items = json.load(open(src, encoding="utf-8"))
    call = make_llm(args.model)
    out, review, dropped = [], [], 0
    for it in items:
        fact = (it.get("source_claim") or "").strip()
        if not fact:
            dropped += 1
            continue
        text = clean(call(PROMPT.format(fact=fact)))
        added_neg = bool(NEG.search(text)) and not NEG.search(fact)
        if not text or text.lower() == fact.lower():
            dropped += 1
            continue
        prefix = "l" if args.tag == "circular_llm" else "m"
        out.append({"id": prefix + it["id"][1:], "pair": it["pair"], "concept": it["concept"],
                    "text": text, "kind": args.tag, "source_claim": fact,
                    "source_fact_id": it.get("source_fact_id"), "rule_circular_id": it["id"],
                    "added_negation_word": added_neg, "generator": args.model,
                    "in_dev55": it.get("in_dev55", False), "dup_cluster": it.get("dup_cluster")})
        review.append({"id": prefix + it["id"][1:], "concept": it["concept"], "source_fact": fact,
                       "llm_error": text, "is_false_given_source (y/n)": ""})
    dst = os.path.join(ROOT, "exp", "data", f"{args.tag}__{args.kb}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    rpath = dst.replace(".json", "__review.csv")
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review[0].keys()))
        w.writeheader()
        w.writerows(review)
    n_neg = sum(o["added_negation_word"] for o in out)
    print(f"{dst}: {len(out)} LLM-written circular errors ({dropped} dropped); "
          f"{n_neg} still added a negation word despite the instruction")
    print(f"hand check: {rpath}  (fill the last column with y/n)")


if __name__ == "__main__":
    main()
