# When AI Evaluates Itself: Code and Data

Code, data, and per-item results for the paper *"When AI Evaluates Itself: Circular Evaluation Bias in Measuring the Reliability of Hallucination Guards for AI Tutoring Systems"*.

The paper studies how the reliability of a knowledge-base (KB) hallucination guard in
an AI tutor is **measured**. It does not propose a new guard. One guard is held fixed
and tested with:

- **5 KBs** built from the same 37 lecture decks: rule-based, sentence split,
  LLM-only, KCA, and KCA re-filtered with MiniCheck
- **6 checkers inside the guard**: RoBERTa-large-MNLI (default), DeBERTa-v3-large,
  ModernBERT-large, two LLM judges (Qwen3.8-Flash, DeepSeek-V4.1-Flash), and a
  frontier LLM (GPT-6 Sol)
- **2 published guards** used whole: MiniCheck-Flan-T5-Large and HHEM-2.1-Open
- **Test sets**: 150 course misconceptions with 136 hand-reviewed correct statements
  (minimal pairs), each KB's circular errors (its own stored negations),
  85 expert misconception/correction pairs from ProgMiscon with an oracle-KB control,
  and FEVER (149 refuted / 148 supported claims)

## Main results

| Result | Value |
|---|---|
| Circular vs independent misconceptions, same 112 concepts (KCA, RoBERTa) | 94.6% vs 59.8% (+34.8 points, p < 0.001) |
| Circular gap, all 17 matched comparisons (every checker, KB, dataset) | +10.1 to +36.6 points; all significant after Holm correction |
| False alarms on the 136 correct statements (KCA) | 28.7% (RoBERTa); 7.4-11.0% (LLM checkers); 67.6-72.8% (MiniCheck, HHEM) |
| KCA vs rule-based KB, paired | +3.3 points, p = 0.50; not significant for any checker |
| ProgMiscon (KCA KB): AUROC, misconception vs its correction | 0.48-0.54 for the NLI-based guards (chance); 0.69-0.70 for LLM checkers |
| Oracle KB (exact correction stored): template vs expert misconception | RoBERTa 100% vs 78.8% (+21.2, p < 0.001); GPT-6 Sol 100% vs 92.9% (p = 0.031) |

## Layout

| Folder / file | What it is |
|---|---|
| `core/guard1_hallucination.py`, `config.py` | The guard under study (its decision rule, Section 3 of the paper) and its thresholds |
| `core/hybrid_extractor.py`, `core/extraction_engine.py`, `core/parsers/` | KB construction: KCA's LLM proposals and entailment gate, the rule-based extraction, the .pptx reader |
| `exp/build_kbs.py` | Builds the course KB arms from the slides (`--regate-model minicheck:flan-t5-large` for the MiniCheck arm) |
| `exp/build_fever.py` | Builds the FEVER test sets and KBs |
| `exp/make_circular.py` | Builds each KB's circular test set |
| `exp/run_eval.py`, `exp/nli_backends.py`, `exp/replay.py` | Instrumented evaluator; swappable checkers (NLI models or LLM judges); offline replay of every verdict |
| `exp/kb_store.py` | File-based KB (one JSONL file per arm) that the guard reads |
| `exp/run_guards.py`, `exp/run_guards.sh`, `exp/analyze_guards.py` | The two published guards on the same KBs and test sets |
| `exp/fetch_progmiscon.py`, `exp/run_progmiscon.sh`, `exp/analyze_progmiscon.py` | ProgMiscon test: download, runs, paired and coverage analysis |
| `exp/make_progmiscon_oracle.py` | Oracle-KB control (ProgMiscon corrections as the KB; expert misconceptions vs template negations) |
| `exp/run_frontier.sh` | Frontier LLM checker (GPT-6 Sol) on the course, ProgMiscon, and oracle tests |
| `exp/run_all.sh` | All course and FEVER experiments in order; resumable |
| `exp/analyze.py` | Statistics (Wilson CIs, exact McNemar, AUROC, catch at FCR <= 5%), tables, ROC figure |
| `exp/data/` | Course misconceptions, correct statements (`true_statements.json` holds the annotators' decisions), circular sets, FEVER sets |
| `exp/kbs/` | All KB arms (JSONL) and their statistics |
| `exp/runs*/` | Per-item logs for every run: every score behind every verdict |
| `exp/results/` | `accepted/summary.json` (main results, 136 accepted correct statements), `guards.json`, `progmiscon/` |
| `exp/cache/` | Cached LLM answers (hash keys only), so reruns are free |
| `experiments/rebuild_log_*.json` | KCA audit trail: every LLM-proposed claim, accepted or rejected |
| `data/annotation/` | The two annotators' reviews of the correct statements |
| `scripts/` | Agreement statistics, figure scripts (`make_paper_figures.py`, `make_overview_figure.py`) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="..."     # only for LLM checkers and KCA; never put keys in files
```

The first run downloads the NLI models (about 1.4-3 GB each). Everything runs on a
laptop CPU.

## Reproduce

```bash
bash exp/run_all.sh                     # course KBs and FEVER; about 12-16 h on a laptop CPU
bash exp/run_guards.sh                  # MiniCheck and HHEM
bash exp/run_progmiscon.sh              # ProgMiscon and the oracle KB (downloads ProgMiscon first)
bash exp/run_frontier.sh                # GPT-6 Sol (about $3 on OpenRouter)
python3 exp/analyze.py --accepted-only --out exp/results/accepted
python3 exp/analyze_guards.py --accepted-only
python3 exp/analyze_progmiscon.py
mkdir -p figures
python3 scripts/make_paper_figures.py exp/results/accepted/summary.json figures
python3 scripts/make_overview_figure.py figures
cp exp/results/accepted/fig_roc.pdf figures/Figure_3.pdf
```

Figures are written as `Figure_1.pdf` ... `Figure_6.pdf` and `Figure_A1.pdf` and follow
Elsevier's artwork rules: 90, 140 or 190 mm wide, Arial-compatible font, no text below
7 pt at final size, vector PDF with embedded fonts.

Every run writes a per-item log that records every score behind every verdict.
`exp/replay.py` recomputes the verdicts from these logs; the replay matched the live
run for every item in every run (0 mismatches), so thresholds, paths, and gates can be
changed without running any model again. All scripts skip finished runs, so they can be
stopped and resumed.

## Data and licences

- **Course slides** (37 decks from two introductory computing courses) are third-party
  course material and are **not** included. The KBs built from them are, so every
  evaluation runs without the slides.
- **FEVER** (Thorne et al., 2018) is CC BY-SA 3.0. To rebuild it, download
  `shared_task_dev.jsonl` and `wiki-pages.zip` from <https://fever.ai/dataset/fever.html>
  and set `FEVER_DEV` and `FEVER_WIKI`.
- **ProgMiscon** (Chiodini et al., 2021; <https://progmiscon.org>) states no licence, so
  its text is not redistributed. `exp/fetch_progmiscon.py` downloads it; the released
  logs keep entry names and scores but not the statement text.
- **Correct statements and course misconceptions** were drafted with LLM help and
  reviewed by hand by two annotators each (see the paper, Section 4.2).

## Notes

- About half of DeepSeek-V4.1-Flash's answers came without log-probabilities and
  GPT-6 Sol's probabilities were almost always 0 or 1, so their AUROC is coarse. Catch
  rates and false-alarm rates are unaffected.
- Only the code the paper uses is included. The original prototype tutor's other components are
  not needed: every KB is a JSONL file in `exp/kbs/`.
