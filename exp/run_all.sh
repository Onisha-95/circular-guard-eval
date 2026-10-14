#!/usr/bin/env bash
# run_all.sh -- every experiment the revised paper needs, in priority order.
# Run from the replication-package root with the venv active:
#     bash exp/run_all.sh            # Tier A + B (about 8-10 h on a laptop CPU)
#     TIER=A bash exp/run_all.sh     # Tier A only (about 3-4 h)
# Re-running skips any run whose output already exists, so it is safe to stop
# and resume. After it finishes:  python3 exp/analyze.py
set -euo pipefail
cd "$(dirname "$0")/.."
TIER="${TIER:-AB}"
RUNS=exp/runs
mkdir -p "$RUNS"

ROBERTA=roberta-large-mnli
DEBERTA=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
MODERNBERT=tasksource/ModernBERT-large-nli          # needs transformers >= 4.48
# LLM judge: OpenRouter if OPENROUTER_API_KEY is set, else OpenAI; override with LLM_JUDGE=llm:<model>[@url]
# Two current LLM judges (Aug/Sep 2026, OpenRouter, log-probs supported). KCA's own LLM stays
# gpt-4o-mini because the course KCA KB was built with it; changing it would confound arms.
OR=https://openrouter.ai/api/v1
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  LLM_JUDGES="${LLM_JUDGES:-llm:qwen/qwen3.8-flash@$OR llm:deepseek/deepseek-v4.1-flash@$OR}"
  KCA_LLM="${KCA_LLM:-openai/gpt-4o-mini@$OR}"
else
  LLM_JUDGES="${LLM_JUDGES:-llm:gpt-4o-mini}"
  KCA_LLM="${KCA_LLM:-gpt-4o-mini}"
fi

ERR=exp/data/errors_noncircular.json
TRUE=exp/data/true_statements_DRAFT.json   # rename once the two annotators have reviewed it

run () {   # run <kb> <items> <model> <retrieval> <tag>
  local out="$RUNS/$5.jsonl"
  if [[ -s "$out" ]]; then echo "skip $5 (exists)"; return; fi
  echo "=== $5"
  python3 exp/run_eval.py --kb "$1" --items "$2" --nli-model "$3" --retrieval "$4" --out "$out.part"
  mv "$out.part" "$out"
}
short () { basename "$1" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-20; }

# ---- 0. KB arms from the same slides (model-free; seconds) -----------------
[[ -s exp/kbs/kca.jsonl ]] || python3 exp/build_kbs.py
for a in baseline_rule sentence_split llm_only kca; do
  [[ -s exp/data/circular__$a.json ]] || python3 exp/make_circular.py --kb exp/kbs/$a.jsonl --out exp/data/circular__$a.json
done

# ---- A. core results: controlled arms, default verifier --------------------
for a in baseline_rule kca sentence_split llm_only; do
  run exp/kbs/$a.jsonl "$ERR"                      $ROBERTA concept "${a}__roberta__concept__errors"
  run exp/kbs/$a.jsonl "$TRUE"                     $ROBERTA concept "${a}__roberta__concept__true"
  run exp/kbs/$a.jsonl exp/data/circular__$a.json  $ROBERTA concept "${a}__roberta__concept__circular"
done

# ---- A. public testbed: FEVER (needs FEVER_DEV and FEVER_WIKI, see exp/build_fever.py) ----
if [[ ! -s exp/kbs/fever_kca.jsonl && -n "${FEVER_DEV:-}" && -n "${FEVER_WIKI:-}" ]]; then
  python3 exp/build_fever.py --dev "$FEVER_DEV" --wiki "$FEVER_WIKI" --llm "$KCA_LLM"
fi
if [[ -s exp/data/fever_errors.json ]]; then
  for a in fever_sentence fever_fragment fever_llm_only fever_kca; do
    [[ -s exp/kbs/$a.jsonl ]] || continue
    [[ -s exp/data/circular__$a.json ]] || python3 exp/make_circular.py --kb exp/kbs/$a.jsonl \
        --concepts exp/data/fever_errors.json --out exp/data/circular__$a.json
    run exp/kbs/$a.jsonl exp/data/fever_errors.json  $ROBERTA concept "${a}__roberta__concept__errors"
    run exp/kbs/$a.jsonl exp/data/fever_true.json    $ROBERTA concept "${a}__roberta__concept__true"
    run exp/kbs/$a.jsonl exp/data/circular__$a.json  $ROBERTA concept "${a}__roberta__concept__circular"
  done
else
  echo "skip FEVER (set FEVER_DEV=shared_task_dev.jsonl FEVER_WIKI=wiki-pages.zip)"
fi

[[ "$TIER" == "A" ]] && { echo "Tier A done. Now: python3 exp/analyze.py"; exit 0; }

# ---- B. retrieval that does not hand the verifier the concept ---------------
for a in baseline_rule kca; do
  run exp/kbs/$a.jsonl "$ERR"  $ROBERTA open "${a}__roberta__open__errors"
  run exp/kbs/$a.jsonl "$TRUE" $ROBERTA open "${a}__roberta__open__true"
done

# ---- B. other verifiers ---------------------------------------------------
# RoBERTa-large-MNLI stays the default because it is the verifier the original guard
# used; these test whether the effects hold for current models.
for m in $DEBERTA $MODERNBERT; do
  s=$(short $m)
  for a in baseline_rule kca; do
    run exp/kbs/$a.jsonl "$ERR"                     $m concept "${a}__${s}__concept__errors"
    run exp/kbs/$a.jsonl "$TRUE"                    $m concept "${a}__${s}__concept__true"
    run exp/kbs/$a.jsonl exp/data/circular__$a.json $m concept "${a}__${s}__concept__circular"
  done
done

# LLMs as the NLI judge (~4k short calls per judge; cached, so reruns are free)
if [[ -n "${OPENAI_API_KEY:-}" || -n "${OPENROUTER_API_KEY:-}" ]]; then
  for LLM in $LLM_JUDGES; do
    s=$(echo "${LLM#llm:}" | cut -d@ -f1 | tr -c 'A-Za-z0-9\n' '_')
    for a in baseline_rule kca; do
      run exp/kbs/$a.jsonl "$ERR"                     "$LLM" concept "${a}__llm_${s}__concept__errors"
      run exp/kbs/$a.jsonl "$TRUE"                    "$LLM" concept "${a}__llm_${s}__concept__true"
      run exp/kbs/$a.jsonl exp/data/circular__$a.json "$LLM" concept "${a}__llm_${s}__concept__circular"
    done
  done
else
  echo "skip LLM judge (set OPENAI_API_KEY, or LLM_JUDGE=llm:<model>@<base_url> for a local server)"
fi

# ---- B. KCA gate with a model other than the verifier -----------------------
# MiniCheck: a purpose-built grounding checker from a different model family.
R=exp/kbs/kca_regate_minicheck_flan_t5_large.jsonl
if [[ ! -s $R ]]; then
  python3 -c "import minicheck" 2>/dev/null || \
    pip install -q accelerate "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
  python3 exp/build_kbs.py --regate-model minicheck:flan-t5-large
fi
[[ -s exp/data/circular__kca_regate.json ]] || \
  python3 exp/make_circular.py --kb $R --out exp/data/circular__kca_regate.json
run $R "$ERR"  $ROBERTA concept "kca_regate__roberta__concept__errors"
run $R "$TRUE" $ROBERTA concept "kca_regate__roberta__concept__true"

echo "All runs done. Now: python3 exp/analyze.py"
