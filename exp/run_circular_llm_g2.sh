#!/usr/bin/env bash
# run_circular_llm_g2.sh -- LLM-written circular errors from a SECOND generator, from a
# different model family that is not used as a checker anywhere in the study.
#
#   export OPENROUTER_API_KEY=...         (in your shell, never in a file)
#   bash exp/run_circular_llm_g2.sh       (GEN2_MODEL=<openrouter id> to choose another model)
#
# Step 1 writes the errors and a review CSV WITHOUT any helper labels. Label it yourself
# (y = clearly false given the source fact, n = otherwise) while steps 2-3 run, then run
#   python3 exp/analyze_circular_llm.py circular_llm_g2
# Resumable; every LLM answer is cached in exp/cache/.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }
OR=https://openrouter.ai/api/v1
GEN2="${GEN2_MODEL:-meta-llama/llama-3.3-70b-instruct}"
QWEN="llm:qwen/qwen3.8-flash@$OR"
TAG=circular_llm_g2

# 0. is the generator available on OpenRouter?
python3 - "$GEN2" <<'EOF'
import json, sys, urllib.request
ids = {m["id"] for m in json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30))["data"]}
m = sys.argv[1]
if m not in ids:
    alt = [i for i in sorted(ids) if any(k in i for k in ("llama-3.3-70b", "llama-4", "gemini", "mistral-small", "mistral-medium"))][:15]
    print(f"{m} is not available on OpenRouter. Pick one of these and rerun with GEN2_MODEL=<id>:")
    print("\n".join("  " + a for a in alt)); sys.exit(1)
print(f"generator: {m}")
EOF

# 1. write the errors (same source facts as before)
for kb in kca baseline_rule; do
  [[ -s exp/data/${TAG}__$kb.json ]] || python3 exp/make_circular_llm.py --kb $kb --model "$GEN2" --tag $TAG
done
echo ">>> Now label exp/data/${TAG}__kca__review.csv and exp/data/${TAG}__baseline_rule__review.csv (y/n) while the runs continue."

# 2. our guard, RoBERTa and Qwen
mkdir -p exp/runs_circllm
for kb in kca baseline_rule; do
  for spec in roberta-large-mnli "$QWEN"; do
    tag=$( [[ "$spec" == roberta* ]] && echo roberta || echo qwen )
    out=exp/runs_circllm/${kb}__${tag}__concept__${TAG}.jsonl
    [[ -s "$out" ]] && { echo "skip $out"; continue; }
    echo "=== $kb / $tag"
    python3 exp/run_eval.py --kb exp/kbs/$kb.jsonl --items exp/data/${TAG}__$kb.json \
      --nli-model "$spec" --retrieval concept --out "$out.part" && mv "$out.part" "$out"
  done
done

# 3. published guards
mkdir -p exp/runs_guards_circllm
for g in minicheck hhem; do
  for kb in kca baseline_rule; do
    out=exp/runs_guards_circllm/${kb}__${g}__${TAG}.jsonl
    [[ -s "$out" ]] && { echo "skip $out"; continue; }
    echo "=== $kb / $g"
    python3 exp/run_guards.py --guard $g --kb exp/kbs/$kb.jsonl \
      --items exp/data/${TAG}__$kb.json --out "$out.part" && mv "$out.part" "$out"
  done
done

echo "Runs done. After labelling both review CSVs:  python3 exp/analyze_circular_llm.py $TAG"
