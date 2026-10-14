"""
kb_store.py -- a file-backed knowledge base that the unmodified verifier can use.

Why this exists
---------------
The original prototype tutor kept its KB in a graph database and a vector index. A KB
here is one JSONL file (one fact per line), so:
  * every construction arm (baseline / sentence-split / LLM-only / KCA / ...)
    is built from the SAME slides into its own file,
  * the exact KB that produced each number can be released.

Fact record (one JSON object per line)
--------------------------------------
  fact_id, concept, claim, negation, is_negation, priority,
  source_file, source_location, arm, [nli_entail_prob]

It implements the two KB methods the verifier calls (get_facts_for_concept,
get_all_facts) plus `retrieve`, which embeds "concept: claim" the same way the
original prototype built its vector index.
"""
import hashlib
import json
import os
import uuid

import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # same embedding model as the guard


def _neo4j_priority_key(f):
    # Neo4j's `ORDER BY f.priority DESC` on strings sorts MEDIUM > LOW > HIGH.
    # We reproduce that ordering so concept-mode retrieval returns facts in
    # the same order as the original evals (it matters for circular items,
    # which take the first stored fact).
    return f.get("priority") or ""


class JsonKB:
    def __init__(self, path, embed_model=None):
        self.path = path
        self.facts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    f = json.loads(line)
                    for k in ("concept", "claim", "negation"):   # Neo4j exports can hold nulls
                        f[k] = f.get(k) or ""
                    f["fact_id"] = f.get("fact_id") or f"row{len(self.facts)}"
                    self.facts.append(f)
        self.by_concept = {}
        for f in self.facts:
            self.by_concept.setdefault((f.get("concept") or "").lower().strip(), []).append(f)
        for lab in self.by_concept:
            # stable sort keeps file order inside ties
            self.by_concept[lab].sort(key=_neo4j_priority_key, reverse=True)
        self._model = embed_model
        self._emb = None

    # --- Neo4jKGClient-compatible surface --------------------------------
    def get_facts_for_concept(self, concept_label):
        out = []
        for f in self.by_concept.get((concept_label or "").lower().strip(), []):
            out.append(self._public(f))
        return out

    def get_all_facts(self):
        return [dict(self._public(f), concept=f.get("concept", "")) for f in self.facts]

    def close(self):
        pass

    @staticmethod
    def _public(f):
        return {
            "fact_id": f["fact_id"],
            "claim": f.get("claim", ""),
            "negation": f.get("negation", ""),
            "is_negation": f.get("is_negation", False),
            "priority": f.get("priority", "MEDIUM"),
            "source_file": f.get("source_file", ""),
            "source_location": f.get("source_location", ""),
            "concept": f.get("concept", ""),
        }

    # --- RAG stand-in -----------------------------------------------------
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
        return self._model

    def _embeddings(self):
        if self._emb is not None:
            return self._emb
        texts = [f"{f.get('concept','')}: {f.get('claim','')}" for f in self.facts]
        digest = hashlib.sha1(("\n".join(texts) + EMBED_MODEL_NAME).encode()).hexdigest()[:12]
        cache = f"{self.path}.{digest}.emb.npy"
        cached = np.load(cache) if os.path.exists(cache) else None
        if cached is not None:
            # guard against a cache written by a different model build
            probe = np.asarray(self._get_model().encode(texts[0], show_progress_bar=False), dtype=np.float32)
            probe /= np.linalg.norm(probe) + 1e-12
            if cached.shape != (len(texts), probe.shape[0]) or float(cached[0] @ probe) < 0.999:
                cached = None
        if cached is not None:
            self._emb = cached
        else:
            m = self._get_model()
            e = m.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            e = np.asarray(e, dtype=np.float32)
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
            try:
                np.save(cache, e)
            except OSError:
                pass
            self._emb = e
        return self._emb

    def retrieve(self, query, top_k=5):
        if not self.facts:
            return []
        e = self._embeddings()
        q = np.asarray(self._get_model().encode(query, show_progress_bar=False), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        sims = e @ q
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            f = self.facts[int(i)]
            r = self._public(f)
            r["similarity"] = round(float(sims[i]), 4)
            out.append(r)
        return out


class RagAdapter:
    """Gives JsonKB the RAGSystem surface the guard / relevance gate expect."""

    def __init__(self, kb):
        self.kb = kb

    def _get_model(self):
        return self.kb._get_model()

    def retrieve_context(self, query, top_k=5):
        return self.kb.retrieve(query, top_k=top_k)


_SEQ = {}


def new_fact(concept, claim, source_file, location, arm, priority="MEDIUM",
             is_negation=False, negation=None, **extra):
    """Build a fact record with the SAME negation rule the original prototype used
    (core/extraction_engine.auto_generate_negation): stored negations are rule-generated
    for every arm, so the verifier's negation paths are identical across arms."""
    from core.extraction_engine import auto_generate_negation
    claim = (claim or "").strip()
    if negation is None:
        negation = claim if is_negation else auto_generate_negation(claim)
    # The original KB kept exact duplicates are kept as separate
    # facts; a per-arm sequence number keeps that behaviour and stays deterministic.
    seq = _SEQ.setdefault(arm, 0)
    _SEQ[arm] = seq + 1
    rec = {
        "fact_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{arm}|{seq}|{source_file}|{location}|{concept}|{claim}")),
        "concept": (concept or "").lower().strip(),
        "claim": claim,
        "negation": negation,
        "is_negation": bool(is_negation),
        "priority": priority,
        "source_file": source_file,
        "source_location": location,
        "arm": arm,
    }
    rec.update(extra)
    return rec


def write_kb(path, facts):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
