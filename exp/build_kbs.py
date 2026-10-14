"""
build_kbs.py -- build every knowledge-base arm from the SAME source slides.

A fair comparison of KB-building methods needs every KB to be built from the same
material. Every arm below reads exactly the same .pptx files (the two courses'
lecture decks; orientation and review decks excluded) and differs ONLY in how
facts are produced:

  baseline_rule   original rule-based ingestion (core/extraction_engine.py),
                  run unmodified through a capture client. OCR off by default
                  (KCA never reads image text); --baseline-ocr turns it on.
  sentence_split  one fact per bullet sentence, slide title as concept.
                  No LLM, no gate. Isolates "atomic + clean concept labels"
                  without any model.
  llm_only        every claim the KCA's LLM proposed (accepted + rejected,
                  from the rebuild logs). Same proposals as KCA, no gate.
  kca             the claims the KCA's NLI gate accepted.
  kca_regate_*    (optional, needs a model) the same proposals re-gated with a
                  DIFFERENT NLI model than the verifier, to test whether the
                  KCA gain depends on gate and verifier sharing roberta-large-mnli.

All arms receive negations from the same rule (auto_generate_negation), exactly
as Neo4jKGClient.create_fact does, so the verifier's negation-based paths are
identical across arms.

USAGE (from the replication-package root):
    python3 exp/build_kbs.py                                  # all model-free arms
    python3 exp/build_kbs.py --regate-model minicheck:flan-t5-large       # recommended
    python3 exp/build_kbs.py --regate-model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
Outputs: exp/kbs/<arm>.jsonl and exp/kbs/kb_stats.json
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from exp.kb_store import new_fact, write_kb  # noqa: E402

EXCLUDED_FILE_PATTERNS = ["welcome", "syllabus", "orientation", "midterm_review", "final_review"]


def course_pptx_files(slides_root, course):
    """Select the course lecture decks (orientation and review decks excluded)."""
    files = glob.glob(os.path.join(slides_root, course, "**", "*.pptx"), recursive=True)
    files = sorted(f for f in files if "__MACOSX" not in f and not os.path.basename(f).startswith("."))
    return [f for f in files if not any(p in os.path.basename(f).lower() for p in EXCLUDED_FILE_PATTERNS)]


# ---------------------------------------------------------------- baseline
class CaptureKG:
    """Records what KnowledgeExtractionEngine would write to Neo4j, with the
    same acceptance rules as Neo4jKGClient (merge_concept / create_fact)."""

    def __init__(self, arm, course):
        self.arm, self.course = arm, course
        self.facts, self.concepts, self.rels = [], set(), 0

    def merge_concept(self, label, source_file, location="", **kw):
        label = (label or "").lower().strip()
        if not label:
            return False
        new = label not in self.concepts
        self.concepts.add(label)
        return new

    def merge_relationship(self, *a, **kw):
        self.rels += 1
        return True

    def create_fact(self, concept_label, claim, source_file, location="",
                    priority="MEDIUM", is_negation=False):
        claim = (claim or "").strip()
        if not claim or len(claim) < 20:          # same rule as neo4j_client.create_fact
            return False
        self.facts.append(new_fact(concept_label, claim, source_file, location, self.arm,
                                   priority=priority, is_negation=is_negation,
                                   course_id=self.course))
        return True

    def create_example(self, *a, **kw):
        return True


def build_baseline(files_by_course, ocr=False):
    from core.parsers import ocr_utils
    if not ocr:
        ocr_utils.TESSERACT_OK = False
    from core.parsers import parse_document
    from core.extraction_engine import KnowledgeExtractionEngine
    facts = []
    for course, files in files_by_course.items():
        cap = CaptureKG("baseline_rule", course)
        eng = KnowledgeExtractionEngine(cap)
        for path in files:
            parsed = parse_document(path)
            parsed["course_id"] = course
            eng.extract_and_write(parsed)
        facts += cap.facts
    return facts


# ---------------------------------------------------------- sentence split
_SENT = re.compile(r"(?<=[.!?])\s+")


def build_sentence_split(files_by_course):
    from core.hybrid_extractor import (extract_pptx_chunks, is_valid_concept_label,
                                       CONCEPT_LABEL_OVERRIDES, clean_text)
    facts = []
    for course, files in files_by_course.items():
        for path in files:
            for ch in extract_pptx_chunks(path):
                if not ch.title or not ch.bullets:
                    continue
                label = ch.title.lower().strip()
                if label in CONCEPT_LABEL_OVERRIDES:
                    label = CONCEPT_LABEL_OVERRIDES[label]
                elif not is_valid_concept_label(ch.title):
                    continue
                for b in ch.bullets:
                    for s in _SENT.split(clean_text(b["text"])):
                        s = s.strip()
                        if len(s) >= 20:
                            facts.append(new_fact(label, s, ch.source_file,
                                                  f"slide {ch.slide_num}", "sentence_split",
                                                  course_id=course))
    return facts


# ------------------------------------------------------- KCA from the logs
def load_logs(log_paths):
    acc, rej = [], []
    for p in log_paths:
        d = json.load(open(p, encoding="utf-8"))
        acc += d.get("accepted", [])
        rej += d.get("rejected", [])
    return acc, rej


def build_from_proposals(records, arm):
    out = []
    for r in records:
        out.append(new_fact(r["concept_label"], r["claim"], r["source_file"], r["location"], arm,
                            priority=r.get("priority", "MEDIUM"), is_negation=False,
                            course_id=r.get("course_id"),
                            nli_entail_prob=r.get("nli_entail_prob")))
    return out


def build_regate(acc, rej, files_by_course, model_name, threshold):
    """Re-gate the identical LLM proposals with another NLI model."""
    from core.hybrid_extractor import (extract_pptx_chunks, build_candidate_premises,
                                       merge_inline_subheaders, SlideChunk, NLIValidator)
    premises = {}
    for files in files_by_course.values():
        for path in files:
            for ch in extract_pptx_chunks(path):
                ch = SlideChunk(slide_num=ch.slide_num, title=ch.title,
                                bullets=merge_inline_subheaders(ch.bullets),
                                source_file=ch.source_file)
                premises[(ch.source_file, f"slide {ch.slide_num}")] = build_candidate_premises(ch)
    if model_name.startswith("minicheck:"):
        # MiniCheck (Tang et al., EMNLP 2024): a grounding checker trained to say whether a
        # document supports a claim -- exactly the KCA gate's question, from a different
        # model family than the RoBERTa verifier. pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
        # MiniCheck loads its model with device_map="auto", which on a laptop short of memory
        # tries to off-load weights to disk and fails. Load on the default device instead.
        import transformers
        for _cls in (transformers.AutoModelForSeq2SeqLM, transformers.AutoModelForSequenceClassification):
            _orig = _cls.from_pretrained
            def _fp(*a, _orig=_orig, **k):
                k.pop("device_map", None)
                return _orig(*a, **k)
            _cls.from_pretrained = _fp
        try:  # MiniCheck splits documents with NLTK's sentence tokenizer
            import nltk
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from minicheck.minicheck import MiniCheck
        mc = MiniCheck(model_name=model_name.split(":", 1)[1],
                       cache_dir=os.path.join(ROOT, "exp", "cache", "minicheck"))

        class _MC:
            def validate_best(self, claim, premises):
                _, probs, _, _ = mc.score(docs=list(premises), claims=[claim] * len(premises))
                i = max(range(len(probs)), key=lambda j: probs[j])
                return probs[i] >= threshold, float(probs[i]), i
        nli = _MC()
    else:
        nli = NLIValidator(model_name=model_name, entail_threshold=threshold)
    arm = "kca_regate_" + re.sub(r"[^a-z0-9]+", "_", model_name.split("/")[-1].lower()).strip("_")
    kept, missing = [], 0
    for r in acc + rej:
        cand = premises.get((r["source_file"], r["location"]))
        if not cand:
            missing += 1
            continue
        ok, prob, _ = nli.validate_best(r["claim"], cand)
        if ok:
            kept.append(dict(r, nli_entail_prob=round(prob, 4)))
    print(f"  re-gate with {model_name}: kept {len(kept)}/{len(acc)+len(rej)} "
          f"(original gate kept {len(acc)}); proposals with no premise found: {missing}")
    return arm, build_from_proposals(kept, arm)


# ------------------------------------------------------------------ stats
def kb_stats(facts):
    facts = [dict(f, claim=f.get("claim") or "", negation=f.get("negation") or "") for f in facts]
    words = [len(f["claim"].split()) for f in facts]
    claims = [re.sub(r"\s+", " ", f["claim"].lower().strip()) for f in facts]
    dup = sum(c - 1 for c in Counter(claims).values() if c > 1)
    multi = sum(1 for f in facts if len(_SENT.split(f["claim"].strip())) > 1 or "\n" in f["claim"])
    return {
        "concepts": len({f["concept"] for f in facts}),
        "facts": len(facts),
        "words_per_fact_median": statistics.median(words) if words else 0,
        "words_per_fact_mean": round(statistics.mean(words), 1) if words else 0,
        "pct_multi_sentence": round(100 * multi / max(len(facts), 1), 1),
        "pct_exact_duplicates": round(100 * dup / max(len(facts), 1), 1),
        "pct_rule_negation_fallback": round(100 * sum(
            f["negation"].startswith("It is NOT the case that:") for f in facts) / max(len(facts), 1), 1),
        "pct_negation_flagged_facts": round(100 * sum(bool(f.get("is_negation")) for f in facts) / max(len(facts), 1), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-root", default=os.path.join(ROOT, "slides"))
    ap.add_argument("--courses", nargs="+", default=["COURSE_A", "COURSE_B"])
    ap.add_argument("--logs", nargs="+", default=[
        os.path.join(ROOT, "experiments", "rebuild_log_COURSE_A.json"),
        os.path.join(ROOT, "experiments", "rebuild_log_COURSE_B.json")])
    ap.add_argument("--out", default=os.path.join(ROOT, "exp", "kbs"))
    ap.add_argument("--baseline-ocr", action="store_true")
    ap.add_argument("--regate-model", default=None)
    ap.add_argument("--regate-threshold", type=float, default=None,
                    help="default 0.65 for NLI classifiers (as KCA), 0.5 for minicheck:* (its own cut-off)")
    args = ap.parse_args()

    files_by_course = {c: course_pptx_files(args.slides_root, c) for c in args.courses}
    acc, rej = load_logs(args.logs)
    logged_files = {r["source_file"] for r in acc + rej}
    used = {os.path.basename(f) for fs in files_by_course.values() for f in fs}
    print(f"Source decks: {len(used)}  (in KCA logs: {len(logged_files)}; "
          f"decks never logged by KCA: {sorted(used - logged_files)})")

    arms = {}
    if args.regate_model:
        thr = args.regate_threshold if args.regate_threshold is not None else (
            0.5 if args.regate_model.startswith("minicheck:") else 0.65)
        arm, facts = build_regate(acc, rej, files_by_course, args.regate_model, thr)
        arms[arm] = facts
    else:
        arms["baseline_rule"] = build_baseline(files_by_course, ocr=args.baseline_ocr)
        arms["sentence_split"] = build_sentence_split(files_by_course)
        arms["llm_only"] = build_from_proposals(acc + rej, "llm_only")
        arms["kca"] = build_from_proposals(acc, "kca")

    stats_path = os.path.join(args.out, "kb_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    for arm, facts in arms.items():
        write_kb(os.path.join(args.out, f"{arm}.jsonl"), facts)
        stats[arm] = kb_stats(facts)
        print(f"  {arm:28s} {stats[arm]}")
    stats["_source"] = {"courses": args.courses, "decks": sorted(used),
                        "baseline_ocr": args.baseline_ocr}
    json.dump(stats, open(stats_path, "w"), indent=2)
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
