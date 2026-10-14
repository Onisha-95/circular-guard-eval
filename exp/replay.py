"""
replay.py -- recompute the verifier's verdict offline from logged pair scores.

run_eval.py logs, for every claim, every score the guard computed
(cosine to each fact, NLI fact->claim contradiction and entailment for the
top-N facts, NLI claim->stored-negation entailment). `claim_verdict` re-applies
the decision rule of core/guard1_hallucination.py::_evaluate_claim line for
line, so any threshold, any subset of contradiction paths, and either relevance
gate can be evaluated without re-running the model. run_eval.py checks that the
replay at the shipped settings reproduces the live verdict for every item.

The guard has three independent ways to output CONTRADICTED:
  nli_fact : NLI says a stored FACT contradicts the claim     (paper's Eq. 1)
  nli_neg  : NLI says the claim ENTAILS a stored NEGATION     (needs negations)
  cosine   : claim embedding is closer to a stored negation than to any fact
"""
ALL_PATHS = ("nli_fact", "nli_neg", "cosine")

DEFAULTS = dict(
    nli_contra_threshold=0.65,   # config.NLI_CONTRA_THRESHOLD
    nli_entail_threshold=0.65,   # config.NLI_ENTAIL_THRESHOLD
    verified_threshold=0.52,     # config.VERIFIED_THRESHOLD
    contradiction_threshold=0.55,  # config.CONTRADICTION_THRESHOLD (cosine path)
    contradiction_gap=0.04,      # config.CONTRADICTION_GAP
    relevance_floor=0.30,        # tau_s, the relevance gate
)


def claim_verdict(c, paths=ALL_PATHS, use_nli=True, **kw):
    p = dict(DEFAULTS, **kw)
    fact_contra, claim_neg, support = 0.0, 0.0, 0.0
    contra_fact = None
    if use_nli:
        for q in c["pairs"]:                       # same order as the guard
            if "nli_fact" in paths and q["contra"] > fact_contra:
                fact_contra, contra_fact = q["contra"], q["fact_id"]
            if q["entail"] > support:
                support = q["entail"]
            if "nli_neg" in paths and q["neg_entail"] is not None and q["neg_entail"] > claim_neg:
                claim_neg = q["neg_entail"]
                if contra_fact is None:
                    contra_fact = q["fact_id"]
    best_nli = max(claim_neg, fact_contra)
    nli_hit = use_nli and best_nli >= p["nli_contra_threshold"] and best_nli > support
    cos_hit = ("cosine" in paths and c["neg_score"] >= p["contradiction_threshold"]
               and c["neg_score"] > c["best_score"] + p["contradiction_gap"])
    if nli_hit or cos_hit:
        f = contra_fact if nli_hit else c["neg_fact_id"]
        f = f or c["neg_fact_id"] or c["best_fact_id"]
        strength = best_nli if nli_hit else c["neg_score"]
        which = ("nli_fact" if (nli_hit and fact_contra >= claim_neg) else
                 "nli_neg" if nli_hit else "cosine")
        return {"verdict": "CONTRADICTED", "fact_id": f, "strength": strength, "path": which,
                "score": best_nli}
    if c["best_score"] >= p["verified_threshold"] or support >= p["nli_entail_threshold"]:
        return {"verdict": "VERIFIED", "fact_id": c["best_fact_id"], "strength": 0.0, "path": None,
                "score": best_nli}
    return {"verdict": "UNVERIFIABLE", "fact_id": c["best_fact_id"], "strength": 0.0, "path": None,
            "score": best_nli}


def item_verdict(rec, gate="fixed", **kw):
    """gate: 'none' (raw), 'fixed' (similarity of the item to the fact that
    actually produced the contradiction) or 'legacy' (similarity to the first
    retrieved fact, as an earlier version of the evaluator did)."""
    p = dict(DEFAULTS, **kw)
    claims = [claim_verdict(c, **kw) for c in rec["claims"]]
    if not claims:
        return {"raw": "VERIFIED", "gated": "VERIFIED", "path": None, "fact_id": None, "score": 0.0}
    vs = [c["verdict"] for c in claims]
    raw = ("CONTRADICTED" if "CONTRADICTED" in vs else
           "VERIFIED" if all(v == "VERIFIED" for v in vs) else "UNVERIFIABLE")
    score = max(c["score"] for c in claims)
    if raw != "CONTRADICTED":
        return {"raw": raw, "gated": raw, "path": None, "fact_id": None, "score": score}
    top = max((c for c in claims if c["verdict"] == "CONTRADICTED"), key=lambda c: c["strength"])
    if gate == "none":
        sim = 1.0
    elif gate == "legacy":
        sim = rec.get("legacy_gate_sim", 1.0)
    else:
        sim = rec["facts"].get(top["fact_id"], {}).get("gate_sim", 1.0) if top["fact_id"] else 1.0
    gated = "CONTRADICTED" if sim >= p["relevance_floor"] else "UNVERIFIABLE"
    return {"raw": raw, "gated": gated, "path": top["path"], "fact_id": top["fact_id"],
            "gate_sim": sim, "score": score}
