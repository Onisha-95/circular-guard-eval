#!/usr/bin/env bash
# Off-the-shelf guards (MiniCheck, HHEM) on the same KBs and test sets as the main study.
# Resumable: finished runs are skipped. Local models only, no API cost.
#   bash exp/run_guards.sh            then   python3 exp/analyze_guards.py
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=exp/runs_guards; mkdir -p "$OUT"
GUARDS="${GUARDS:-minicheck hhem}"

run () {  # run <guard> <kb-name> <items> <set>
  local o="$OUT/$2__$1__$4.jsonl"
  [[ -s "$o" ]] && { echo "skip $o"; return; }
  echo "=== $2 / $1 / $4"
  python3 exp/run_guards.py --guard "$1" --kb "exp/kbs/$2.jsonl" --items "$3" --out "$o"
}

for g in $GUARDS; do
  for kb in baseline_rule kca; do                                   # course testbed
    run $g $kb exp/data/errors_noncircular.json errors
    run $g $kb exp/data/true_statements_DRAFT.json true
    run $g $kb exp/data/circular__$kb.json circular
  done
  for kb in fever_sentence fever_kca; do                            # FEVER, if built
    [[ -s exp/kbs/$kb.jsonl && -s exp/data/fever_errors.json ]] || continue
    run $g $kb exp/data/fever_errors.json errors
    run $g $kb exp/data/fever_true.json true
    run $g $kb exp/data/circular__$kb.json circular
  done
done
echo "All guard runs done. Now: python3 exp/analyze_guards.py"
