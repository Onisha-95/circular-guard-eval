"""
nli_backends.py -- plug a different NLI judge into the unmodified guard.

The guard asks two questions of its NLI model (core/guard1_hallucination.py):
  _nli_scores(premise, hypothesis) -> {entailment, neutral, contradiction}
  _nli_entail(premise, hypothesis) -> P(entailment)
Any backend that answers those with 3-way probabilities can stand in for
roberta-large-mnli without touching the decision rule.

Spec strings accepted by run_eval.py --nli-model:
  <hf model id>          a 3-way NLI classifier from the Hugging Face hub
                         (roberta-large-mnli, MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli,
                          tasksource/ModernBERT-large-nli, ...). Loaded by the guard's own pipeline.
  llm:<model>            an instruction-tuned LLM used as a 3-way NLI judge through any
                         OpenAI-compatible endpoint. Probabilities come from the log-probs of
                         the one-letter answer. Examples:
                           llm:gpt-4o-mini                              (OPENAI_API_KEY)
                           llm:openai/gpt-4o-mini@https://openrouter.ai/api/v1 (OPENROUTER_API_KEY)
                           llm:qwen2.5:7b-instruct@http://localhost:11434/v1   (Ollama, no key)
                         Every answer is cached in exp/cache/, so reruns cost nothing.

Binary grounding checkers (MiniCheck, FactCG, Granite Guardian) output only
"supported / not supported". They cannot tell a contradicted claim from an
unrelated one, so they are NOT valid here; use them for the KCA gate instead
(exp/build_kbs.py --regate-model minicheck:...).
"""
import hashlib
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("entailment", "neutral", "contradiction")


# ------------------------------------------------------------ HF classifiers
def check_hf_labels(guard):
    """The guard reads labels by NAME. A checkpoint whose config says LABEL_0/1/2
    would silently score 0 everywhere and make the guard cosine-only, so refuse."""
    nli = guard._get_nli()
    if nli is None:
        raise SystemExit("NLI model failed to load.")
    out = nli({"text": "A dog is running in the park.", "text_pair": "An animal is outside."})
    if out and isinstance(out[0], list):
        out = out[0]
    names = {d["label"].lower() for d in out}
    missing = [l for l in LABELS if l not in names]
    if missing:
        raise SystemExit(f"NLI model labels are {sorted(names)}; missing {missing}. "
                         f"Pick a checkpoint whose id2label names entailment/neutral/contradiction.")
    # the guard swallows exceptions and returns zeros; make sure it really gets scores
    sc = guard._nli_scores("A dog is running in the park.", "An animal is outside.")
    if sum(sc.values()) < 0.5:
        raise SystemExit("The guard's NLI call returned no scores with this model/transformers "
                         "version (it fails silently). Check the pipeline output format.")


# ------------------------------------------------------------------- LLM judge
PROMPT = (
    "Decide the relation between the PREMISE and the HYPOTHESIS.\n"
    "E = the premise entails the hypothesis (if the premise is true, the hypothesis must be true)\n"
    "C = the premise contradicts the hypothesis (they cannot both be true)\n"
    "N = neither (unrelated, or the premise does not settle it)\n\n"
    "PREMISE: {p}\nHYPOTHESIS: {h}\n\nAnswer with one letter: E, C or N."
)


def with_retries(call, attempts=8):
    """Retry rate limits, timeouts and provider 5xx errors with exponential backoff
    (2, 4, 8, ... up to 120 s). A 400 (bad request) is not retried: it is a real bug."""
    import time
    import openai
    for k in range(attempts):
        try:
            return call()
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                openai.InternalServerError) as e:
            if k == attempts - 1:
                raise
            wait = min(120, 2 ** (k + 1))
            print(f"    [retry {k + 1}/{attempts - 1} in {wait}s: {type(e).__name__}]", flush=True)
            time.sleep(wait)
        except openai.APIStatusError as e:
            if getattr(e, "status_code", 0) in (408, 409, 429, 500, 502, 503, 504) and k < attempts - 1:
                wait = min(120, 2 ** (k + 1))
                print(f"    [retry {k + 1}/{attempts - 1} in {wait}s: HTTP {e.status_code}]", flush=True)
                time.sleep(wait)
                continue
            raise


class LLMJudge:
    def __init__(self, spec):
        body = spec.split(":", 1)[1]
        self.model, _, self.base_url = body.partition("@")
        self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL") or None
        from openai import OpenAI
        if self.base_url and "openrouter" in self.base_url:
            key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        else:
            key = os.environ.get("OPENAI_API_KEY") or ("ollama" if self.base_url else None)
        if not key:
            raise SystemExit("Set OPENAI_API_KEY (or give an @base_url for a local server).")
        self.client = OpenAI(api_key=key, base_url=self.base_url)
        os.makedirs(os.path.join(ROOT, "exp", "cache"), exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", spec)
        self.cache_path = os.path.join(ROOT, "exp", "cache", f"nli__{safe}.jsonl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                d = json.loads(line)
                self.cache[d["k"]] = d["v"]
        self.calls = self.hits = 0
        self.no_logprobs = 0

    def scores(self, premise, hypothesis):
        k = hashlib.sha1(f"{self.model}\x00{premise}\x00{hypothesis}".encode()).hexdigest()
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self.calls += 1
        extra = {}
        if self.base_url and "openrouter" in self.base_url:
            # current OpenRouter models are often hybrid "thinking" models; ask for a direct answer
            extra = {"reasoning": {"enabled": False}}
        r = with_retries(lambda: self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=8, logprobs=True, top_logprobs=5,
            messages=[{"role": "user", "content": PROMPT.format(p=premise, h=hypothesis)}],
            extra_body=extra or None))
        ch = r.choices[0]
        mass = {"E": 0.0, "N": 0.0, "C": 0.0}
        lp = getattr(ch, "logprobs", None)
        if lp and lp.content:
            # use the first generated token that is the answer letter (skips a leading space/newline)
            pos = next((t for t in lp.content if t.token.strip().upper()[:1] in mass), lp.content[0])
            for t in (pos.top_logprobs or []):
                tok = t.token.strip().upper()[:1]
                if tok in mass:
                    mass[tok] += math.exp(t.logprob)
        if sum(mass.values()) == 0:          # server gave no log-probs: use the answer itself
            self.no_logprobs += 1
            tok = (ch.message.content or "N").strip().upper()[:1]
            mass[tok if tok in mass else "N"] = 1.0
        z = sum(mass.values())
        v = {"entailment": mass["E"] / z, "neutral": mass["N"] / z, "contradiction": mass["C"] / z}
        self.cache[k] = v
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "v": v}) + "\n")
        return v


def install(guard, spec):
    """Point the guard at the requested NLI backend. Returns a status callable."""
    if spec.startswith("llm:"):
        judge = LLMJudge(spec)
        guard.use_nli = True
        guard._get_nli = lambda: judge
        guard._nli_scores = lambda p, h: judge.scores(p, h) if p and h else dict.fromkeys(LABELS, 0.0)
        guard._nli_entail = lambda p, h: judge.scores(p, h)["entailment"] if p and h else 0.0
        return lambda: (f"LLM judge {judge.model}: {judge.calls} calls, {judge.hits} cache hits"
                        + (f", {judge.no_logprobs} answers without log-probs (one-hot)" if judge.no_logprobs else ""))
    import core.guard1_hallucination as G
    G.NLI_MODEL_NAME = spec
    check_hf_labels(guard)
    return lambda: f"HF NLI {spec}"
