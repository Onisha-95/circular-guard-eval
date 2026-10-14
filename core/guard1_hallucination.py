"""
Guard Agent #1 — Hallucination Detection.

Validates an LLM response claim by claim against the stored facts of a knowledge
base (in this release, a JSONL file read through exp/kb_store.py).
Uses sentence-transformer similarity + explicit negation matching.

Verdicts:
  VERIFIED     — all claims match KG facts (similarity >= verified_threshold)
  CONTRADICTED — negation match exceeds positive match by contradiction_gap
  UNVERIFIABLE — claims present but no KG fact matches above threshold

FIX (BUG 4): _evaluate_claim() rewritten.
  Old version set verdict="CONTRADICTED" inside the loop before the gap check,
  then re-evaluated outside — the best_fact pointer was wrong when contradiction
  lost the gap check, causing missed real contradictions and false positives.
  New version tracks best_fact and neg_fact separately, resolves verdict once
  at the end with clean priority ordering.
"""
import re, time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import (VERIFIED_THRESHOLD, CONTRADICTION_THRESHOLD,
                        CONTRADICTION_GAP)
except ImportError:
    VERIFIED_THRESHOLD      = 0.52
    CONTRADICTION_THRESHOLD = 0.58
    CONTRADICTION_GAP       = 0.04

# NLI settings (W-1, proper sentence-level integration).
# NLI is run claim-by-claim, sentence-to-sentence, in BOTH directions:
#   - claim ENTAILS a stored negation  -> contradiction signal
#   - a stored fact ENTAILS the claim  -> support (verified) signal
# NLI owns the CONTRADICTED decision at its natural threshold; cosine remains
# the VERIFIED/fallback signal. To keep sentence-level NLI affordable, only the
# top-N cosine-closest facts are NLI-checked per claim.
try:
    from config import (USE_NLI_CONTRADICTION, NLI_MODEL_NAME,
                        NLI_CONTRA_THRESHOLD, NLI_ENTAIL_THRESHOLD,
                        NLI_TOP_FACTS)
except ImportError:
    USE_NLI_CONTRADICTION = True
    NLI_MODEL_NAME        = "facebook/bart-large-mnli"
    NLI_CONTRA_THRESHOLD  = 0.65   # claim->negation entailment prob to flag contradiction
    NLI_ENTAIL_THRESHOLD  = 0.65   # fact->claim entailment prob to count as support
    NLI_TOP_FACTS         = 5      # how many cosine-closest facts to NLI-check per claim


class HallucinationDetectionAgent:

    def __init__(self, neo4j_client, rag_system,
                 verified_threshold=VERIFIED_THRESHOLD,
                 contradiction_threshold=CONTRADICTION_THRESHOLD,
                 contradiction_gap=CONTRADICTION_GAP,
                 use_nli=USE_NLI_CONTRADICTION,
                 nli_contra_threshold=NLI_CONTRA_THRESHOLD,
                 nli_entail_threshold=NLI_ENTAIL_THRESHOLD,
                 nli_top_facts=NLI_TOP_FACTS):
        self.kg                      = neo4j_client
        self.rag                     = rag_system
        self.verified_threshold      = verified_threshold
        self.contradiction_threshold = contradiction_threshold
        self.contradiction_gap       = contradiction_gap
        self.use_nli                 = use_nli
        self.nli_contra_threshold    = nli_contra_threshold
        self.nli_entail_threshold    = nli_entail_threshold
        self.nli_top_facts           = nli_top_facts
        self._model                  = None
        self._nli                    = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _get_nli(self):
        """Lazy-load the NLI pipeline once. Returns None if disabled or load
        fails, so the guard always falls back cleanly to cosine-only."""
        if not self.use_nli:
            return None
        if self._nli is None:
            try:
                from transformers import pipeline
                self._nli = pipeline("text-classification",
                                     model=NLI_MODEL_NAME,
                                     top_k=None, truncation=True)
            except Exception:
                self.use_nli = False
                self._nli = None
        return self._nli

    def _nli_entail(self, premise: str, hypothesis: str) -> float:
        """Probability that `premise` ENTAILS `hypothesis`, sentence-level.
        bart-large-mnli takes the (premise, hypothesis) pair; we read the
        'entailment' label. Returns 0.0 on any failure or empty input."""
        nli = self._get_nli()
        if nli is None or not premise or not hypothesis:
            return 0.0
        try:
            out = nli({"text": premise, "text_pair": hypothesis})
            scores = {d["label"].lower(): d["score"] for d in out}
            return scores.get("entailment", 0.0)
        except Exception:
            return 0.0

    def _nli_scores(self, premise: str, hypothesis: str) -> dict:
        """All three NLI label probabilities for the ordered pair
        (premise -> hypothesis): entailment / neutral / contradiction.
        This is what lets us read the CONTRADICTION label directly, instead of
        relying on a stored negation string. Returns zeros on any failure."""
        nli = self._get_nli()
        zero = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        if nli is None or not premise or not hypothesis:
            return zero
        try:
            out = nli({"text": premise, "text_pair": hypothesis})
            return {d["label"].lower(): d["score"] for d in out}
        except Exception:
            return zero

    # -- MAIN VALIDATE METHOD -----------------------------------------------
    def validate(self, llm_response: str, retrieved_facts: list,
                 query: str = "") -> dict:
        t0 = time.time()

        claims = self._split_claims(llm_response)
        if not claims:
            return self._build_result(
                "VERIFIED", [], [], [],
                llm_response, t0,
                note="No verifiable claims detected."
            )

        # Expand fact pool: RAG facts + Neo4j facts for concepts in response
        all_facts = list(retrieved_facts)
        concepts_in_response = self._extract_concepts(llm_response)
        for concept in concepts_in_response:
            neo4j_facts = self.kg.get_facts_for_concept(concept)
            existing_ids = {f.get("fact_id") for f in all_facts}
            for f in neo4j_facts:
                if f.get("fact_id") not in existing_ids:
                    all_facts.append(f)

        model         = self._get_model()
        claim_results = []
        flagged       = []
        corrections   = []

        for claim in claims:
            verdict, best_score, best_fact, neg_score, diag = self._evaluate_claim(
                claim, all_facts, model
            )
            claim_results.append({
                "claim":     claim,
                "verdict":   verdict,
                "score":     round(best_score, 4),
                "neg_score": round(neg_score, 4),
                "fact_id":   best_fact.get("fact_id", "") if best_fact else "",
                "nli":       diag,
            })
            if verdict in ("CONTRADICTED", "UNVERIFIABLE"):
                flagged.append(claim)
            if verdict == "CONTRADICTED" and best_fact:
                correct = best_fact.get("claim",
                          best_fact.get("f.claim", "See course materials."))
                source  = best_fact.get("source_location",
                          best_fact.get("f.source_location", ""))
                corrections.append({
                    "claim":      claim,
                    "correction": correct,
                    "source":     source,
                })

        verdicts = [r["verdict"] for r in claim_results]
        if "CONTRADICTED" in verdicts:
            overall = "CONTRADICTED"
        elif all(v == "VERIFIED" for v in verdicts):
            overall = "VERIFIED"
        else:
            overall = "UNVERIFIABLE"

        delivered = self._build_delivery(overall, llm_response, corrections)
        return self._build_result(
            overall, claim_results, flagged, corrections, delivered, t0
        )

    # -- EVALUATE ONE CLAIM -------------------------------------------------
    def _evaluate_claim(self, claim: str, facts: list, model):
        """
        Two-stage claim evaluation:
          Stage 1 (cosine, cheap, all facts): find the closest positive fact and
                  closest negation, exactly as before. Cosine remains the VERIFIED
                  signal and the fallback when NLI is disabled.
          Stage 2 (NLI, precise, top-N cosine-closest facts only): run sentence-
                  level entailment in BOTH directions on the most relevant facts:
                    - claim ENTAILS stored negation  -> contradiction evidence
                    - stored fact ENTAILS claim       -> support evidence
                  NLI owns the CONTRADICTED decision at its natural threshold.
        """
        from sentence_transformers import util

        claim_emb  = model.encode(claim, convert_to_tensor=True)
        best_score = 0.0
        best_fact  = None
        neg_score  = 0.0
        neg_fact   = None
        scored     = []   # (pos_sim, fact) for ranking which facts to NLI-check

        for fact in facts:
            claim_text = fact.get("claim",    fact.get("f.claim",    ""))
            neg_text   = fact.get("negation", fact.get("f.negation", ""))
            if not claim_text:
                continue

            fact_emb = model.encode(claim_text, convert_to_tensor=True)
            pos_sim  = util.cos_sim(claim_emb, fact_emb).item()
            scored.append((pos_sim, fact))
            if pos_sim > best_score:
                best_score = pos_sim
                best_fact  = fact

            if neg_text:
                n_emb  = model.encode(neg_text, convert_to_tensor=True)
                n_sim  = util.cos_sim(claim_emb, n_emb).item()
                if n_sim > neg_score:
                    neg_score = n_sim
                    neg_fact  = fact

        # ---- Stage 2: sentence-level NLI on the top-N cosine-closest facts ----
        nli_contra_score = 0.0   # best claim->negation entailment (legacy signal)
        nli_fact_contra  = 0.0   # NEW: best stored-fact->claim CONTRADICTION prob
        nli_contra_fact  = None
        nli_support      = 0.0   # best fact->claim entailment (support)
        if self.use_nli and scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, fact in scored[: self.nli_top_facts]:
                claim_text = fact.get("claim",    fact.get("f.claim",    ""))
                neg_text   = fact.get("negation", fact.get("f.negation", ""))
                # NEW primary signal: does the stored FACT contradict the claim?
                # Reads the NLI 'contradiction' label directly, so it catches
                # novel/paraphrased misconceptions with no matching stored negation.
                if claim_text:
                    sc = self._nli_scores(claim_text, claim)
                    if sc["contradiction"] > nli_fact_contra:
                        nli_fact_contra = sc["contradiction"]
                        nli_contra_fact = fact
                    if sc["entailment"] > nli_support:
                        nli_support = sc["entailment"]
                # legacy signal kept: does the claim entail the stored negation?
                if neg_text:
                    c = self._nli_entail(claim, neg_text)
                    if c > nli_contra_score:
                        nli_contra_score = c
                        if nli_contra_fact is None:
                            nli_contra_fact = fact

        # ---- Verdict resolution ----
        # CONTRADICTED if NLI is confident the claim entails a negation AND the
        # claim is not simultaneously well-supported (fact entails claim). This
        # bidirectional check is what prevents the false-contradiction blow-up.
        # Use the STRONGER of the two contradiction signals:
        #   nli_fact_contra  : stored fact contradicts the claim (NEW, general)
        #   nli_contra_score : claim entails stored negation       (legacy)
        best_nli_contra = max(nli_contra_score, nli_fact_contra)
        nli_contra = (
            self.use_nli
            and best_nli_contra >= self.nli_contra_threshold
            and best_nli_contra > nli_support
        )
        # Cosine contradiction kept as a secondary trigger (clear negation win).
        cosine_contra = (neg_score >= self.contradiction_threshold
                         and neg_score > best_score + self.contradiction_gap)

        diag = {
            "nli_fact_contra":  round(nli_fact_contra, 4),   # NEW: fact contradicts claim
            "nli_claim_neg":    round(nli_contra_score, 4),  # legacy: claim entails negation
            "nli_support":      round(nli_support, 4),       # fact entails claim
            "cosine_neg":       round(neg_score, 4),
            "cosine_pos":       round(best_score, 4),
        }
        if nli_contra or cosine_contra:
            contra_fact = nli_contra_fact if nli_contra else neg_fact
            return ("CONTRADICTED", best_score,
                    contra_fact or neg_fact or best_fact,
                    max(neg_score, nli_contra_score, nli_fact_contra), diag)
        elif best_score >= self.verified_threshold or nli_support >= self.nli_entail_threshold:
            return "VERIFIED", best_score, best_fact, neg_score, diag
        elif (neg_score >= self.contradiction_threshold
              and neg_score > best_score):
            return "UNVERIFIABLE", best_score, best_fact, neg_score, diag
        else:
            return "UNVERIFIABLE", best_score, best_fact, neg_score, diag

    # -- HELPERS ------------------------------------------------------------
    def _split_claims(self, text: str) -> list:
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if 15 < len(s.strip()) < 400]

    def _extract_concepts(self, text: str) -> list:
        words  = re.findall(r"\b[a-z][a-z ]{2,25}\b", text.lower())
        common = {"the", "and", "for", "this", "that", "with", "you",
                  "can", "will", "are", "was", "from", "they", "have"}
        return list(set(w.strip() for w in words if w.strip() not in common))[:15]

    def _build_delivery(self, verdict: str, original: str,
                        corrections: list) -> str:
        if verdict == "CONTRADICTED":
            clean_corrections = []
            for c in corrections:
                text = (c.get("correction") or "").strip()
                if text and text not in clean_corrections:
                    clean_corrections.append(text)
            if clean_corrections:
                return (
                    " ".join(clean_corrections)
                    + "\n\n📌 Answer verified and corrected against your course slides."
                )
            return original
        elif verdict == "UNVERIFIABLE":
            return (
                original + "\n\n"
                "⚠️ This answer could not be fully verified in course slides yet."
            )
        else:
            return original

    def _build_result(self, verdict, claim_results, flagged,
                      corrections, delivered, t0, note="") -> dict:
        return {
            "verdict":            verdict,
            "claim_results":      claim_results,
            "flagged_claims":     flagged,
            "corrections":        corrections,
            "delivered_response": delivered,
            "note":               note,
            "latency_ms":         round((time.time() - t0) * 1000, 2),
        }
