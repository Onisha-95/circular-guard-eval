#!/usr/bin/env bash
# MiniCheck and HHEM scored fact by fact (the setting most favourable to them), on both
# course KBs and every test set. Local models only; finished sets are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
for g in minicheck hhem; do
  for kb in kca baseline_rule; do
    python3 exp/run_guards_perfact.py --guard $g --kb $kb
  done
done
