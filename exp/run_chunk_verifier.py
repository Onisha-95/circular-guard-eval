"""
run_chunk_verifier.py -- a chunk-level support verifier (paper, Appendix B).

A design described for a recent programming tutor: course material cut into
180-250-word chunks; hybrid retrieval (sentence-transformers/all-mpnet-base-v2 +
BM25, reciprocal-rank fusion), top 3 chunks per claim; a DeBERTa-v3-large NLI
model; a claim is supported if one chunk entails it with probability >= 0.65,
otherwise it is flagged. That system's code is not public and its paper does not
name the checkpoint or the premise order, so this is a verifier of the general
design, NOT a reproduction of that system. Chunks are built from each KB's facts
in slide order (the slides are not redistributed). Both premise orders are run.

Measured as in the main study: catch rate, FCR on the accepted correct statements,
AUROC, catch at FCR <= 5%, and matched circular gaps (rule-made and hand-checked
LLM-written), exact McNemar.

USAGE
  pip install rank_bm25
  python3 exp/run_chunk_verifier.py --kb kca                       (chunk is premise)
  python3 exp/run_chunk_verifier.py --kb kca --order claim_first   (claim is premise)
Outputs: exp/runs_chunk_verifier[_claimfirst]/<kb>__<set>.jsonl,
         exp/results/chunk_verifier__<kb>[_claimfirst].json
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENTAIL = 0.65
TOP_K = 3
MIN_W, MAX_W = 180, 250


def toks(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


# ------------------------------------------------------------------ chunks
def slide_no(f):
    m = re.search(r"(\d+)", f.get("source_location") or "")
    return int(m.group(1)) if m else 0


def build_chunks(kb_path):
    facts = [json.loads(l) for l in open(kb_path, encoding="utf-8") if l.strip()]
    facts = [f for f in facts if (f.get("claim") or "").strip()
             and str(f.get("is_negation", "False")).lower() != "true"]
    facts.sort(key=lambda f: (f.get("source_file") or "", slide_no(f)))
    chunks, cur, n, deck, seen = [], [], 0, None, set()
    for f in facts:
        c = f["claim"].strip()
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        if deck is not None and f.get("source_file") != deck and n >= MIN_W // 2:
            chunks.append(" ".join(cur)); cur, n = [], 0
        deck = f.get("source_file")
        w = len(c.split())
        if n and n + w > MAX_W:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(c); n += w
        if n >= MIN_W:
            chunks.append(" ".join(cur)); cur, n = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------- retrieval
class Hybrid:
    def __init__(self, chunks, stub=False):
        self.chunks = chunks
        self.stub = stub
        if stub:
            self.sets = [set(toks(c)) for c in chunks]
            return
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        self.bm25 = BM25Okapi([toks(c) for c in chunks])
        self.emb = SentenceTransformer(EMB_MODEL)
        self.mat = self.emb.encode(chunks, normalize_embeddings=True, show_progress_bar=True,
                                   batch_size=32)

    def top(self, q, k=TOP_K, rrf_k=60):
        if self.stub:
            t = set(toks(q))
            sc = [len(t & s) / (len(t | s) or 1) for s in self.sets]
            return list(np.argsort(sc)[::-1][:k])
        dense = self.mat @ self.emb.encode([q], normalize_embeddings=True)[0]
        sparse = self.bm25.get_scores(toks(q))
        fused = np.zeros(len(self.chunks))
        for scores in (dense, sparse):
            for rank, i in enumerate(np.argsort(scores)[::-1][:100]):
                fused[i] += 1.0 / (rrf_k + rank + 1)
        return list(np.argsort(fused)[::-1][:k])


# ---------------------------------------------------------------------- NLI
def make_nli(stub=False):
    if stub:
        def entail(premise, hyp):
            if re.search(r"\bnot\b|n't", hyp.lower()):
                return 0.05
            t = set(toks(hyp))
            return len(t & set(toks(premise))) / max(len(t), 1)
        return entail
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval().to(dev)
    print(f"NLI on {dev}")
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    ei = lab["entailment"]

    def entail(premise, hyp):
        x = tok(premise, hyp, truncation="longest_first", max_length=512, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**x).logits[0].float(), -1)
        return float(p[ei])
    return entail


# ----------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)]


def mcnemar(b, c):
    from scipy import stats
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def auroc(pos, neg):
    from scipy import stats
    if not pos or not neg:
        return None
    return round(float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic
                       / (len(pos) * len(neg))), 3)


def catch_at(pos, neg, max_fcr=0.05):
    best = 0.0
    for t in sorted(set(pos) | set(neg) | {1.01}):
        if np.mean([s >= t for s in neg]) <= max_fcr:
            best = max(best, float(np.mean([s >= t for s in pos])))
    return round(100 * best, 1)


def rate(recs):
    k, n = sum(r["flagged"] for r in recs), len(recs)
    return {"k": k, "n": n, "rate": round(100 * k / n, 1) if n else None, "ci": wilson(k, n)}


def gap(err, circ):
    ef = {r["id"]: r["flagged"] for r in err}
    pairs = [(ef[c["pair"]], c["flagged"]) for c in circ if c.get("pair") in ef]
    if not pairs:
        return None
    b = sum(1 for e, c in pairs if e and not c)
    c_ = sum(1 for e, c in pairs if c and not e)
    pe, pc = np.mean([e for e, _ in pairs]), np.mean([c for _, c in pairs])
    return {"n": len(pairs), "independent": round(100 * pe, 1), "circular": round(100 * pc, 1),
            "gap": round(100 * (pc - pe), 1), "p": mcnemar(b, c_)}


def reviewed_out(tag, kb):
    """ids marked n (not clearly false) in the hand-check file of an LLM-written set"""
    import csv
    p = os.path.join(ROOT, "exp", "data", f"{tag}__{kb}__review.csv")
    if not os.path.exists(p):
        return set()
    return {r["id"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))
            if (r.get("is_false_given_source (y/n)") or "").strip().lower().startswith("n")}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="kca or baseline_rule")
    ap.add_argument("--stub", action="store_true", help="pipeline test without models")
    ap.add_argument("--order", choices=["chunk_first", "claim_first"], default="chunk_first",
                    help="NLI premise: the evidence chunk (default) or the claim")
    args = ap.parse_args()

    kb_path = os.path.join(ROOT, "exp", "kbs", f"{args.kb}.jsonl")
    chunks = build_chunks(kb_path)
    wl = [len(c.split()) for c in chunks]
    print(f"{args.kb}: {len(chunks)} chunks, median {int(np.median(wl))} words "
          f"(range {min(wl)}-{max(wl)})")
    ret = Hybrid(chunks, stub=args.stub)
    entail = make_nli(stub=args.stub)

    data = os.path.join(ROOT, "exp", "data")
    keep = {it["id"] for it in json.load(open(os.path.join(data, "true_statements.json"), encoding="utf-8"))
            if it.get("review") == "kept"}
    sets = {"errors": json.load(open(os.path.join(data, "errors_noncircular.json"), encoding="utf-8")),
            "true": [it for it in json.load(open(os.path.join(data, "true_statements.json"), encoding="utf-8"))
                     if it["id"] in keep],
            "circular": json.load(open(os.path.join(data, f"circular__{args.kb}.json"), encoding="utf-8"))}
    lp = os.path.join(data, f"circular_llm__{args.kb}.json")
    if os.path.exists(lp):
        sets["circular_llm"] = json.load(open(lp, encoding="utf-8"))

    suffix = "" if args.order == "chunk_first" else "_claimfirst"
    outdir = os.path.join(ROOT, "exp", "runs_chunk_verifier" + suffix + ("_stub" if args.stub else ""))
    os.makedirs(outdir, exist_ok=True)
    runs = {}
    for name, items in sets.items():
        path = os.path.join(outdir, f"{args.kb}__{name}.jsonl")
        if os.path.exists(path):
            runs[name] = [json.loads(l) for l in open(path, encoding="utf-8")][1:]
            print(f"  reuse {path}")
            continue
        t0, recs = time.time(), []
        with open(path + ".part", "w", encoding="utf-8") as out:
            out.write(json.dumps({"_meta": True, "design": "chunk-level support verifier", "kb": args.kb,
                                  "nli": "stub" if args.stub else NLI_MODEL, "emb": EMB_MODEL,
                                  "top_k": TOP_K, "entail_threshold": ENTAIL,
                                  "chunks": len(chunks), "order": args.order}) + "\n")
            for n, it in enumerate(items, 1):
                idx = ret.top(it["text"])
                ps = [entail(chunks[i], it["text"]) if args.order == "chunk_first"
                      else entail(it["text"], chunks[i]) for i in idx]
                best = max(ps) if ps else 0.0
                rec = {k: it.get(k) for k in ("id", "pair", "kind", "concept", "text")}
                rec.update(chunk_ids=[int(i) for i in idx], entail=[round(p, 4) for p in ps],
                           support=round(best, 4), flagged=best < ENTAIL)
                recs.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if n % 25 == 0 or n == len(items):
                    print(f"  {name}: {n}/{len(items)}  {(time.time() - t0) / n:.2f}s/item", flush=True)
        os.replace(path + ".part", path)
        runs[name] = recs

    err, tru = runs["errors"], runs["true"]
    pos = [1 - r["support"] for r in err]
    neg = [1 - r["support"] for r in tru]
    res = {"kb": args.kb, "order": args.order, "chunks": len(chunks), "catch": rate(err), "fcr": rate(tru),
           "auroc": auroc(pos, neg), "catch_at_fcr5": catch_at(pos, neg),
           "circular_rule": gap(err, runs["circular"]),
           "circular_llm": gap(err, [r for r in runs["circular_llm"]
                                     if r["id"] not in reviewed_out("circular_llm", args.kb)])
           if "circular_llm" in runs else None}
    os.makedirs(os.path.join(ROOT, "exp", "results"), exist_ok=True)
    rp = os.path.join(ROOT, "exp", "results", f"chunk_verifier__{args.kb}{suffix}" + ("_stub" if args.stub else "") + ".json")
    json.dump(res, open(rp, "w"), indent=1)

    print(f"\n=== chunk-level support verifier ({args.order}), {args.kb} KB ===")
    print(f"catch on 150 misconceptions:      {res['catch']['rate']}%  CI {res['catch']['ci']}")
    print(f"FCR on {res['fcr']['n']} correct statements:   {res['fcr']['rate']}%  CI {res['fcr']['ci']}")
    print(f"AUROC {res['auroc']}   catch at FCR<=5%: {res['catch_at_fcr5']}%")
    for k in ("circular_rule", "circular_llm"):
        g = res[k]
        if g:
            print(f"{k:14s} n={g['n']}  independent {g['independent']}%  circular {g['circular']}%  "
                  f"gap {g['gap']:+.1f}  p={g['p']:.3g}")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
