#!/usr/bin/env bash
# ProgMiscon (independent, expert-curated misconceptions) on the course KBs.
# Open retrieval: ProgMiscon's concept names differ from the slide titles.
#   bash exp/run_progmiscon.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -s exp/data/progmiscon_errors.json ]] || python3 exp/fetch_progmiscon.py
E=exp/data/progmiscon_errors.json; T=exp/data/progmiscon_true.json
R=exp/runs_progmiscon; G=exp/runs_guards_progmiscon; mkdir -p $R $G
OR=https://openrouter.ai/api/v1
run () {  # run <kb> <model> <short> <set> <items>
  local o="$R/$1__$3__open__$4.jsonl"
  [[ -s "$o" ]] && { echo "skip $o"; return; }
  echo "=== $1 / $3 / $4"
  python3 exp/run_eval.py --kb exp/kbs/$1.jsonl --items "$5" --nli-model "$2" --retrieval open --out "$o.part"
  mv "$o.part" "$o"
}
grun () {  # grun <kb> <guard> <set> <items>
  local o="$G/$1__$2__$3.jsonl"
  [[ -s "$o" ]] && { echo "skip $o"; return; }
  echo "=== $1 / $2 / $3"
  python3 exp/run_guards.py --guard $2 --kb exp/kbs/$1.jsonl --items "$4" --retrieval open --out "$o"
}
for kb in baseline_rule kca; do
  run $kb roberta-large-mnli roberta errors $E
  run $kb roberta-large-mnli roberta true   $T
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    run $kb "llm:qwen/qwen3.8-flash@$OR" qwen errors $E
    run $kb "llm:qwen/qwen3.8-flash@$OR" qwen true   $T
  fi
  for g in minicheck hhem; do
    grun $kb $g errors $E
    grun $kb $g true   $T
  done
done
# Oracle control: the KB holds the exact correction of every misconception.
# Expert-written errors vs template negations of the same 85 facts.
[[ -s exp/kbs/progmiscon_oracle.jsonl ]] || python3 exp/make_progmiscon_oracle.py
OK=progmiscon_oracle; TP=exp/data/progmiscon_oracle_template.json
R=exp/runs_progmiscon_oracle; G=exp/runs_guards_progmiscon_oracle; mkdir -p $R $G
run $OK roberta-large-mnli roberta errors   $E
run $OK roberta-large-mnli roberta template $TP
for g in minicheck hhem; do
  grun $OK $g errors   $E
  grun $OK $g template $TP
done
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  run $OK "llm:qwen/qwen3.8-flash@$OR" qwen errors   $E
  run $OK "llm:qwen/qwen3.8-flash@$OR" qwen template $TP
fi
echo "Done. Now:"
echo "  python3 exp/analyze_progmiscon.py"
echo "  python3 exp/analyze.py --runs exp/runs_progmiscon --out exp/results/progmiscon"
echo "  python3 exp/analyze_guards.py --runs exp/runs_guards_progmiscon --out exp/results/progmiscon/guards.json"
