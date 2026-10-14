"""
Hybrid Knowledge Extractor — LLM-proposes -> source-validate (NLI) -> atomic.

Replaces the regex-only pipeline (extraction_engine.py) for PPTX sources.
Design goals, directly targeting the KG-quality misses found in diagnosis:
  - one atomic claim per fact (no sentence-blob facts, no garbled fragments)
  - every claim is source-validated (NLI entailment) against the slide it
    came from, before being accepted -> filters hallucinated/added content
  - junk filter rejects quiz questions, instructions-to-student, and
    non-factual fragments before they ever reach the KG

OUTPUT SHAPE matches Neo4jKGClient.create_fact() inputs exactly:
    {
        "concept_label": str,
        "claim": str,
        "source_file": str,
        "location": str,       # e.g. "slide 4"
        "priority": "HIGH"|"MEDIUM",
        "is_negation": bool,   # always False here; auto_generate_negation()
                                # in create_fact() derives the negation
    }

THIS MODULE WRITES NOTHING TO NEO4J. It only proposes candidate facts.
Call site decides whether/how to write them (see dry_run_extractor.py).

Requires (on the machine actually running this — NOT the sandbox that
wrote it): python-pptx, openai, transformers, torch.
    pip install python-pptx
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field


# Zero-width / BOM characters found embedded in real slide text (confirmed
# in COURSE_A-internet-networkig.pptx, slide 15 — content pasted from Word/web
# into PowerPoint carries these invisibly). They degrade both embedding
# similarity and NLI tokenization without being visible to a human reviewer.
_INVISIBLE_CHARS = re.compile(
    "[\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\ufffe]"
)

def clean_text(text: str) -> str:
    """Strip BOM/zero-width characters and normalize unicode. Applied to
    every chunk of extracted text before it ever reaches the LLM or NLI."""
    if not text:
        return text
    text = _INVISIBLE_CHARS.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return text


# -----------------------------------------------------------------------
# STEP 0 — PPTX -> slide chunks (tested against real course files)
# -----------------------------------------------------------------------

@dataclass
class SlideChunk:
    slide_num: int
    title: str
    bullets: list = field(default_factory=list)   # [{"text": str, "bold": bool}]
    source_file: str = ""

    def as_text_block(self) -> str:
        """Flatten chunk into plain text for the LLM prompt and NLI premise."""
        lines = [self.title or ""]
        for b in self.bullets:
            lines.append(b["text"])
        return "\n".join(l for l in lines if l.strip())


def extract_pptx_chunks(path: str, source_file: str = None) -> list:
    """One chunk per slide: title + bullet paragraphs with bold flags.
    Verified against real COURSE_B/COURSE_A slides — clean, no garbling.
    """
    from pptx import Presentation
    prs = Presentation(path)
    source_file = source_file or path.split("/")[-1]
    chunks = []
    for i, slide in enumerate(prs.slides):
        title = None
        bullets = []
        shapes = list(slide.shapes)
        for idx, shape in enumerate(shapes):
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                txt = clean_text("".join(r.text for r in para.runs).strip())
                if not txt:
                    continue
                is_bold = any(r.font.bold for r in para.runs if r.font.bold is not None)
                if title is None and idx == 0:
                    title = txt
                else:
                    bullets.append({"text": txt, "bold": is_bold})
        chunks.append(SlideChunk(slide_num=i + 1, title=title or "", bullets=bullets,
                                  source_file=source_file))
    return chunks


_INLINE_LABEL_WHITELIST = {
    "python", "java", "c#", "c++", "javascript", "html", "css", "sql",
    "in both languages", "both languages", "in python", "in java", "in c#",
}

def merge_inline_subheaders(bullets: list) -> list:
    """Detect short standalone bullets that act as inline mini-headers
    (e.g. 'Python', 'Java', 'In both languages') and prepend them as
    context to subsequent bullets, so claims don't lose which language
    they apply to.

    Found via real data: the 'math operators - multiplication' slide
    stores 'Python' and 'Java' as their own bullets, with the actual
    claims as separate bullets right after — without this merge, the
    extractor produces two unlabeled, seemingly-contradictory claims
    ('you can multiply a string by an integer' / 'you may not multiply
    strings with numbers') under the same concept with no way to tell
    which language each is about.
    """
    out = []
    current_label = None
    for b in bullets:
        text = b["text"].strip()
        stripped = text.rstrip(":").lower()
        is_known_label = stripped in _INLINE_LABEL_WHITELIST
        # Generic fallback: short, single-token, capitalized, no punctuation —
        # catches language names not in the whitelist (Ruby, Kotlin, etc.)
        # without hardcoding every possible one.
        is_generic_label = (
            len(text.split()) == 1
            and len(text) <= 12
            and text[:1].isupper()
            and text.replace("#", "").replace("+", "").isalpha()
        )
        if is_known_label or is_generic_label:
            current_label = text.rstrip(":")
            continue  # the label itself isn't a claim candidate
        if current_label:
            out.append({"text": f"{current_label}: {text}", "bold": b["bold"]})
        else:
            out.append(b)
    return out


# -----------------------------------------------------------------------
# STEP 1 — LLM proposes atomic claims from a chunk
# -----------------------------------------------------------------------

EXTRACTION_PROMPT = """You are extracting verifiable factual claims from one slide of a college lecture, for a fact-checking knowledge base.

SLIDE CONTENT:
---
{slide_text}
---

Extract every distinct factual claim that is DIRECTLY stated on this slide. Rules:
- Each claim must be a complete, standalone sentence a fact-checker could verify against this slide alone.
- Do NOT add outside knowledge, examples, or elaboration not present on the slide.
- Do NOT include: quiz/exam questions, instructions to students ("complete the following"), slide labels/headers, agenda items, or pure code with no accompanying natural-language claim.
- Split compound bullets into separate atomic claims (one fact per claim).
- If a bullet is just a heading, example list, or has no checkable claim, skip it.
- Preserve the slide's own wording closely — do not paraphrase technical terms.

Return ONLY a JSON array, no other text:
[{{"claim": "...", "priority": "HIGH"|"MEDIUM"}}]

Use "HIGH" priority only for claims that were bolded or marked important on the slide.
If there are no extractable claims, return [].
"""


def propose_claims_llm(chunk: SlideChunk, llm_call_fn) -> list:
    """llm_call_fn(prompt: str) -> str   (caller supplies the actual API call,
    e.g. a lambda wrapping Neo4jKGClient.call_llm with your an LLM API key).
    Returns list of {"claim": str, "priority": str} — UNVALIDATED.
    """
    slide_text = chunk.as_text_block()
    if not slide_text.strip():
        return []
    prompt = EXTRACTION_PROMPT.format(slide_text=slide_text)
    raw = llm_call_fn(prompt)
    raw = raw.strip()
    # Strip markdown fences if the model added them despite instructions
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if isinstance(item, dict) and item.get("claim"):
            out.append({
                "claim": str(item["claim"]).strip(),
                "priority": item.get("priority", "MEDIUM") if item.get("priority") in
                            ("HIGH", "MEDIUM") else "MEDIUM",
            })
    return out


# -----------------------------------------------------------------------
# STEP 2 — Atomic / junk filter (cheap, no model calls)
# -----------------------------------------------------------------------

_QUESTION_STARTS = (
    "what", "which", "why", "how", "when", "where", "who", "select",
    "choose", "fill in", "true or false", "match the", "circle the",
)

CONCEPT_LABEL_OVERRIDES = {
    # original slide title (lowercased, stripped) -> clean concept label to use instead.
    # Found via real run: "good algorithms are:" got caught by the trailing-colon
    # junk filter even though it's a genuinely useful topic — and its facts are
    # the exact corrective content for the "importance of algorithmic thinking"
    # miss (ambiguity/correctness). Add more entries here as you review skipped
    # titles in the rebuild log and find other false positives.
    "good algorithms are:": "algorithm properties",
}


def is_valid_concept_label(label: str) -> bool:
    """Slide titles are messier than the regex-extractor's old heading
    sources — they include sentence fragments, formatting artifacts, and
    banner text that make terrible concept nodes. Found via real data:
    '**.. which shares the same letters as algorithm', '(from back of
    shampoo bottle)', 'in computing:', 'finally...'. None of these are
    topics; they're title-slide noise. Also reuses extraction_engine's
    existing _valid() to catch the established junk patterns (generic
    words like 'summary', date strings, etc.) so both pipelines agree.
    """
    if not label:
        return False
    t = label.strip()
    if not t[0].isalnum():             # leading *, (, -, bullet glyphs etc.
        return False
    if t.endswith((":", "...", "…")):  # sentence-fragment-as-heading pattern
        return False
    if len(t.split()) > 7:             # likely a full sentence, not a topic
        return False
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from extraction_engine import _valid
        return _valid(t)
    except ImportError:
        return True  # if extraction_engine isn't importable, don't block on it


def is_junk_claim(claim: str) -> bool:
    c = claim.strip()
    if len(c) < 20 or len(c) > 300:
        return True
    if c.endswith("?"):
        return True
    low = c.lower()
    if any(low.startswith(s) for s in _QUESTION_STARTS):
        return True
    # mostly symbols/code, no natural-language content
    alpha_ratio = sum(ch.isalpha() for ch in c) / max(len(c), 1)
    if alpha_ratio < 0.4:
        return True
    return False


def dedupe_claims(claims: list) -> list:
    """Drop near-identical claims (case/whitespace-insensitive exact match).
    Cheap safety net; the LLM prompt already discourages duplicates.
    """
    seen = set()
    out = []
    for c in claims:
        key = re.sub(r"\s+", " ", c["claim"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# -----------------------------------------------------------------------
# STEP 3 — Source-validate via NLI entailment (slide_text -> claim)
# -----------------------------------------------------------------------

class NLIValidator:
    """Wraps roberta-large-mnli. Confirms the slide text actually entails
    the proposed claim, before the claim is allowed into the KG.
    This is the gate that stops LLM-invented content from being written.
    """
    def __init__(self, model_name: str = "roberta-large-mnli", entail_threshold: float = 0.65):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.threshold = entail_threshold
        # id2label varies by checkpoint; resolve the entailment index by label name
        id2label = {k: v.upper() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, lbl in id2label.items() if "ENTAIL" in lbl)

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                 truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self.torch.softmax(logits, dim=-1)[0]
        return float(probs[self.entail_idx])

    def validate(self, claim: str, source_text: str) -> tuple:
        """Returns (is_valid: bool, entail_prob: float)."""
        prob = self.entailment_prob(premise=source_text, hypothesis=claim)
        return prob >= self.threshold, prob

    def validate_best(self, claim: str, candidate_premises: list) -> tuple:
        """Checks the claim against each candidate premise separately and
        takes the BEST score, instead of one diluted whole-slide premise.

        Why this exists: validating against the full slide block as a single
        premise was found to be inconsistent — a claim quoted VERBATIM from
        an early bullet scored as low as 0.04 entailment, while an equally
        verbatim claim from the last bullet scored 0.98, purely due to
        position/dilution in the combined premise. Checking bullet-by-bullet
        (or small windows) removes that bias.

        Returns (is_valid, best_prob, best_premise_index).
        """
        best_prob, best_idx = 0.0, -1
        for i, premise in enumerate(candidate_premises):
            prob = self.entailment_prob(premise=premise, hypothesis=claim)
            if prob > best_prob:
                best_prob, best_idx = prob, i
        return best_prob >= self.threshold, best_prob, best_idx


# -----------------------------------------------------------------------
# ORCHESTRATION — propose -> filter -> validate -> atomic fact dicts
# -----------------------------------------------------------------------

def build_candidate_premises(chunk: SlideChunk) -> list:
    """Title + each individual bullet, PLUS adjacent bullet pairs (so claims
    that combine two consecutive bullets still have a fair premise to match
    against). Avoids validating against the whole slide as one diluted block.
    """
    title = chunk.title or ""
    bullets = [b["text"] for b in chunk.bullets]
    candidates = [f"{title}\n{b}" for b in bullets]
    for i in range(len(bullets) - 1):
        candidates.append(f"{title}\n{bullets[i]}\n{bullets[i+1]}")
    if not candidates:
        candidates = [title]
    return candidates


def extract_chunk_to_facts(chunk: SlideChunk, llm_call_fn, nli_validator: NLIValidator,
                            course_id: str = None, module_name: str = None) -> dict:
    """Returns {"accepted": [...], "rejected": [...]} for one slide.
    Each accepted item is ready to pass to kg.create_fact(**fact) once
    you've eyeballed it (this function does NOT write to Neo4j).
    """
    concept_label = (chunk.title or "").lower().strip()
    location = f"slide {chunk.slide_num}"
    accepted, rejected = [], []

    if concept_label in CONCEPT_LABEL_OVERRIDES:
        concept_label = CONCEPT_LABEL_OVERRIDES[concept_label]
    elif not is_valid_concept_label(chunk.title or ""):
        return {"accepted": [], "rejected": [],
                "skipped_title": chunk.title, "skipped_location": location}

    # Merge inline language sub-headers (Python/Java/C# bullets) into the
    # claims that follow them, so language context isn't lost — see
    # merge_inline_subheaders() docstring for the real case this fixes.
    chunk = SlideChunk(slide_num=chunk.slide_num, title=chunk.title,
                        bullets=merge_inline_subheaders(chunk.bullets),
                        source_file=chunk.source_file)

    proposed = propose_claims_llm(chunk, llm_call_fn)
    proposed = [p for p in proposed if not is_junk_claim(p["claim"])]
    proposed = dedupe_claims(proposed)

    candidate_premises = build_candidate_premises(chunk)
    for item in proposed:
        is_valid, prob, best_idx = nli_validator.validate_best(item["claim"], candidate_premises)
        record = {
            "concept_label": concept_label,
            "claim": item["claim"],
            "source_file": chunk.source_file,
            "location": location,
            "priority": item["priority"],
            "is_negation": False,
            "course_id": course_id,
            "module_name": module_name,
            "nli_entail_prob": round(prob, 4),
        }
        if is_valid:
            accepted.append(record)
        else:
            rejected.append(record)

    return {"accepted": accepted, "rejected": rejected}
