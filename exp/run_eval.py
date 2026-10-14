"""
run_eval.py -- one instrumented evaluator for every table in the paper.

It runs the UNMODIFIED decision logic of core/guard1_hallucination.py (the
class is subclassed only to record what it computes) and writes one JSON line
per test item with every score the verdict depended on. exp/analyze.py then
derives catch rate, false-contradiction rate, threshold sweeps, path
ablations, both relevance gates, overlap and fact-length analyses from these
logs without re-running any model.

Design notes:
  * the relevance gate is computed on the fact that ACTUALLY produced the
    contradiction (an earlier version used the first retrieved fact); the
    older value is logged too, as legacy_gate_sim;
  * retrieval can be the concept-label lookup (--retrieval concept) or open retrieval from the item text alone
    (--retrieval open), which does not hand the verifier the right concept;
  * any NLI model can be swapped in (--nli-model) for the verifier study.

USAGE (from the replication-package root):
  python3 exp/run_eval.py --kb exp/kbs/kca.jsonl --items exp/data/errors_noncircular.json \
         --out exp/runs/kca__roberta__concept__errors.jsonl
  --nli-model tasksource/ModernBERT-large-nli          # any 3-way NLI classifier
  --nli-model llm:gpt-4o-mini                          # an LLM as the NLI judge (see nli_backends.py)
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import numpy as np  # noqa: E402

import core.guard1_hallucination as G  # noqa: E402
from exp.replay import item_verdict  # noqa: E402
from exp.nli_backends import install as install_backend  # noqa: E402


# --------------------------------------------------------------- the guard
class InstrumentedGuard(G.HallucinationDetectionAgent):
    """Identical decisions; records every intermediate score.

    _evaluate_claim below is a line-for-line copy of the parent method with
    `trace` bookkeeping added. run_eval.py asserts after every item that the
    recorded scores replay to the same verdict the method returned."""

    def validate(self, llm_response, retrieved_facts, query=""):
        self._trace = []
        return super().validate(llm_response, retrieved_facts, query)

    def _evaluate_claim(self, claim, facts, model):
        from sentence_transformers import util

        claim_emb = model.encode(claim, convert_to_tensor=True)
        best_score, best_fact, neg_score, neg_fact = 0.0, None, 0.0, None
        scored = []
        for fact in facts:
            claim_text = fact.get("claim", fact.get("f.claim", ""))
            neg_text = fact.get("negation", fact.get("f.negation", ""))
            if not claim_text:
                continue
            fact_emb = model.encode(claim_text, convert_to_tensor=True)
            pos_sim = util.cos_sim(claim_emb, fact_emb).item()
            scored.append((pos_sim, fact))
            if pos_sim > best_score:
                best_score, best_fact = pos_sim, fact
            if neg_text:
                n_emb = model.encode(neg_text, convert_to_tensor=True)
                n_sim = util.cos_sim(claim_emb, n_emb).item()
                if n_sim > neg_score:
                    neg_score, neg_fact = n_sim, fact

        pairs = []
        nli_contra_score = nli_fact_contra = nli_support = 0.0
        nli_contra_fact = None
        if self.use_nli and scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            for pos_sim, fact in scored[: self.nli_top_facts]:
                claim_text = fact.get("claim", fact.get("f.claim", ""))
                neg_text = fact.get("negation", fact.get("f.negation", ""))
                rec = {"fact_id": fact.get("fact_id", ""), "cos_pos": round(pos_sim, 4),
                       "contra": 0.0, "entail": 0.0, "neutral": 0.0, "neg_entail": None}
                if claim_text:
                    sc = self._nli_scores(claim_text, claim)
                    rec.update(contra=round(sc["contradiction"], 4), entail=round(sc["entailment"], 4),
                               neutral=round(sc.get("neutral", 0.0), 4))
                    if sc["contradiction"] > nli_fact_contra:
                        nli_fact_contra, nli_contra_fact = sc["contradiction"], fact
                    if sc["entailment"] > nli_support:
                        nli_support = sc["entailment"]
                if neg_text:
                    c = self._nli_entail(claim, neg_text)
                    rec["neg_entail"] = round(c, 4)
                    if c > nli_contra_score:
                        nli_contra_score = c
                        if nli_contra_fact is None:
                            nli_contra_fact = fact
                pairs.append(rec)
                self._seen[rec["fact_id"]] = fact

        best_nli_contra = max(nli_contra_score, nli_fact_contra)
        nli_contra = (self.use_nli and best_nli_contra >= self.nli_contra_threshold
                      and best_nli_contra > nli_support)
        cosine_contra = (neg_score >= self.contradiction_threshold
                         and neg_score > best_score + self.contradiction_gap)
        diag = {"nli_fact_contra": round(nli_fact_contra, 4), "nli_claim_neg": round(nli_contra_score, 4),
                "nli_support": round(nli_support, 4), "cosine_neg": round(neg_score, 4),
                "cosine_pos": round(best_score, 4)}
        if nli_contra or cosine_contra:
            contra_fact = nli_contra_fact if nli_contra else neg_fact
            out = ("CONTRADICTED", best_score, contra_fact or neg_fact or best_fact,
                   max(neg_score, nli_contra_score, nli_fact_contra), diag)
        elif best_score >= self.verified_threshold or nli_support >= self.nli_entail_threshold:
            out = ("VERIFIED", best_score, best_fact, neg_score, diag)
        else:
            out = ("UNVERIFIABLE", best_score, best_fact, neg_score, diag)

        for f in (best_fact, neg_fact, out[2]):
            if f:
                self._seen[f.get("fact_id", "")] = f
        self._trace.append({
            "claim": claim, "pool_size": len(scored),
            "best_score": round(best_score, 4), "best_fact_id": best_fact.get("fact_id") if best_fact else None,
            "neg_score": round(neg_score, 4), "neg_fact_id": neg_fact.get("fact_id") if neg_fact else None,
            "pairs": pairs, "online_verdict": out[0],
            "online_fact_id": out[2].get("fact_id") if out[2] else None,
        })
        return out


class ExpansionSwitch:
    """The guard enlarges its fact pool with facts of every concept label that
    appears as a word sequence in the response. --no-expand disables that by
    giving the guard a KB view that returns nothing for those lookups."""

    def __init__(self, kb, enabled):
        self.kb, self.enabled = kb, enabled

    def get_facts_for_concept(self, label):
        return self.kb.get_facts_for_concept(label) if self.enabled else []

    def __getattr__(self, name):
        return getattr(self.kb, name)


# ------------------------------------------------------------- KB backends
def open_kb(spec):
    from exp.kb_store import JsonKB, RagAdapter
    kb = JsonKB(spec)
    return kb, RagAdapter(kb), kb._get_model


def load_items(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):            # legacy {concept: wrong_answer}
        return [{"id": f"e{i:03d}", "concept": c, "text": t, "kind": "error"}
                for i, (c, t) in enumerate(d.items())]
    return d


def tokens(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def jaccard(a, b):
    A, B = set(tokens(a)), set(tokens(b))
    return len(A & B) / len(A | B) if A | B else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="exp/kbs/<arm>.jsonl")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nli-model", default=G.NLI_MODEL_NAME)
    ap.add_argument("--retrieval", choices=["concept", "open"], default="concept")
    ap.add_argument("--open-k", type=int, default=10)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    kb, rag, get_model = open_kb(args.kb)
    guard = InstrumentedGuard(ExpansionSwitch(kb, not args.no_expand), rag)
    backend_status = install_backend(guard, args.nli_model)   # HF classifier or llm:<model>
    emb = get_model()

    def esim(a, b):
        va, vb = emb.encode([a, b], show_progress_bar=False)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

    items = load_items(args.items)[: args.limit]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = {"_meta": True, "kb": args.kb, "items": args.items, "nli_model": args.nli_model,
            "retrieval": args.retrieval, "open_k": args.open_k, "expand": not args.no_expand,
            "thresholds": {k: getattr(guard, k) for k in (
                "verified_threshold", "contradiction_threshold", "contradiction_gap",
                "nli_contra_threshold", "nli_entail_threshold", "nli_top_facts")},
            "started": datetime.now().isoformat()}
    mismatches = 0
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps(meta) + "\n")
        for n, it in enumerate(items, 1):
            text, concept = it["text"], it.get("concept", "")
            fallback = False
            if args.retrieval == "concept":
                retrieved = kb.get_facts_for_concept(concept)
                n_concept = len(retrieved)
                if len(retrieved) < 3:                       # concept retrieval: add embedding retrieval when < 3 facts
                    fallback = True
                    have = {f.get("fact_id") for f in retrieved}
                    for f in rag.retrieve_context(f"Can you explain {concept} to me?", top_k=5):
                        if f.get("fact_id") not in have:
                            retrieved.append(f)
            else:
                n_concept = len(kb.get_facts_for_concept(concept))
                retrieved = rag.retrieve_context(text, top_k=args.open_k)
            rec = dict(it, n_concept_facts=n_concept, fallback=fallback,
                       retrieved_ids=[f.get("fact_id") for f in retrieved], claims=[], facts={})
            if not retrieved:
                rec["online"] = "NO_FACTS"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            guard._seen = {}
            res = guard.validate(text, retrieved, query=concept)
            rec["online"] = res["verdict"]
            rec["claims"] = guard._trace
            first = (retrieved[0].get("claim") or "").strip()
            rec["legacy_gate_sim"] = round(esim(text, first), 4) if first else 1.0
            for fid, f in guard._seen.items():
                ftxt = f.get("claim", "")
                rec["facts"][fid] = {
                    "claim": ftxt, "concept": f.get("concept", ""),
                    "words": len(ftxt.split()), "gate_sim": round(esim(text, ftxt), 4),
                    "jaccard": round(jaccard(text, ftxt), 4),
                    "neg_jaccard": round(jaccard(text, f.get("negation", "")), 4)}
            replay = item_verdict(rec, gate="none")["raw"]
            if replay != rec["online"]:
                mismatches += 1
                rec["replay_mismatch"] = replay
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 10 == 0 or n == len(items):
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {el/n:.1f}s/item  replay mismatches: {mismatches}", flush=True)
    print(f"Saved {args.out}   [{backend_status()}]")
    if mismatches:
        print(f"WARNING: {mismatches} items where the offline replay disagrees with the guard. "
              f"Do not use replay-derived numbers until this is understood.")


if __name__ == "__main__":
    main()
