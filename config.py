# Settings of the hallucination guard studied in the paper (core/guard1_hallucination.py).
# exp/replay.py repeats these defaults so verdicts can be recomputed from the run logs.

# Cosine-similarity path (all-MiniLM-L6-v2 embeddings)
VERIFIED_THRESHOLD      = 0.52   # claim counts as verified if similarity to a fact >= this
CONTRADICTION_THRESHOLD = 0.55   # tau_n: similarity to a stored negation needed for path (c)
CONTRADICTION_GAP       = 0.04   # delta: negation must beat the best fact by this margin

# NLI paths (a) and (b)
USE_NLI_CONTRADICTION = True
NLI_MODEL_NAME        = "roberta-large-mnli"   # default verifier; others are swapped in by exp/nli_backends.py
NLI_CONTRA_THRESHOLD  = 0.65   # tau_c
NLI_ENTAIL_THRESHOLD  = 0.65   # fact -> claim entailment that counts as support
NLI_TOP_FACTS         = 5      # facts per claim sent to the NLI model
