#!/usr/bin/env bash
# run_circular_llm.sh -- does the circular gap survive when the circular errors are
# written by an LLM (minimal edits of the KB's own facts, the way published
# hallucination test sets are often built) instead of the KB's rule-made negations?
#
#   export OPENROUTER_API_KEY=...      (in your shell, never in a file)
#   bash exp/run_circular_llm.sh
#
# Resumable: finished runs are skipped and every LLM answer is cached in exp/cache/.
# Cost: about 230 short gpt-4o-mini calls to write the errors, plus the Qwen judge
# runs (roughly 2,000-3,000 short calls). RoBERTa, MiniCheck and HHEM run locally.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }
OR=https://openrouter.ai/api/v1
QWEN="llm:qwen/qwen3.8-flash@$OR"

# 1. write the LLM circular errors (same source facts as the rule-made circular set)
for kb in kca baseline_rule; do
  [[ -s exp/data/circular_llm__$kb.json ]] || python3 exp/make_circular_llm.py --kb $kb
done

# 2. our guard, RoBERTa (local) and Qwen (API), concept retrieval as in the main study
mkdir -p exp/runs_circllm
for kb in kca baseline_rule; do
  for spec in roberta-large-mnli "$QWEN"; do
    tag=$( [[ "$spec" == roberta* ]] && echo roberta || echo qwen )
    out=exp/runs_circllm/${kb}__${tag}__concept__circular_llm.jsonl
    [[ -s "$out" ]] && { echo "skip $out"; continue; }
    echo "=== $kb / $tag"
    python3 exp/run_eval.py --kb exp/kbs/$kb.jsonl --items exp/data/circular_llm__$kb.json \
      --nli-model "$spec" --retrieval concept --out "$out.part" && mv "$out.part" "$out"
  done
done

# 3. published guards (local)
mkdir -p exp/runs_guards_circllm
for g in minicheck hhem; do
  for kb in kca baseline_rule; do
    out=exp/runs_guards_circllm/${kb}__${g}__circular_llm.jsonl
    [[ -s "$out" ]] && { echo "skip $out"; continue; }
    echo "=== $kb / $g"
    python3 exp/run_guards.py --guard $g --kb exp/kbs/$kb.jsonl \
      --items exp/data/circular_llm__$kb.json --out "$out.part" && mv "$out.part" "$out"
  done
done

python3 exp/analyze_circular_llm.py
