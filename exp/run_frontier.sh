#!/usr/bin/env bash
# run_frontier.sh -- one current frontier LLM as the verifier, on the tests that matter most:
#   1. course, KCA KB: misconceptions, correct statements, circular errors (the circular gap)
#   2. ProgMiscon, KCA KB, open retrieval: misconceptions and corrections
#   3. ProgMiscon oracle KB: expert misconceptions vs template negations
# Same prompt and decision rule as the other LLM judges. Resumable: finished runs are
# skipped and every answer is cached in exp/cache/, so a stopped run costs nothing to redo.
#   bash exp/run_frontier.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OR=https://openrouter.ai/api/v1
MODEL="${FRONTIER_MODEL:-openai/gpt-6-sol}"
SPEC="llm:$MODEL@$OR"
TAG=frontier
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }

# --- budget check (reads only your key's usage and limit) ---------------------
python3 - <<'EOF'
import json, os, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                             headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
lim, used = d.get("limit"), d.get("usage", 0)
left = None if lim is None else lim - used
print(f"OpenRouter key: used ${used:.2f}" + ("" if lim is None else f" of ${lim:.2f}  ->  ${left:.2f} left"))
if left is not None and left < 3.5:
    print("Less than $3.50 left. The full run needs about $3-3.5.")
    print("Raise the key's limit on openrouter.ai (Settings > Keys), or press Enter to run what fits.")
EOF

# --- smoke test: 3 known pairs; stops if the model gives no usable answers -------
python3 - "$SPEC" <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
from exp.nli_backends import LLMJudge
j = LLMJudge(sys.argv[1])
tests = [("A while loop checks its condition before each iteration.",
          "A while loop runs its body once before checking the condition.", "contradiction"),
         ("Python lists are mutable.", "You can change the items of a Python list.", "entailment"),
         ("Python lists are mutable.", "HTTP sends data in plain text.", "neutral")]
ok = 0
for p, h, want in tests:
    s = j.scores(p, h)
    got = max(s, key=s.get)
    ok += got == want
    print(f"  want {want:13s} got {got:13s} {s}")
print(f"smoke test: {ok}/3 correct, {j.no_logprobs} answers without log-probs")
if ok < 2:
    sys.exit("The model is not answering usably (reasoning may be on, or the id is wrong). Stopping.")
EOF

run () {  # run <kb> <items> <retrieval> <out>
  [[ -s "$4" ]] && { echo "skip $4"; return; }
  echo "=== $4"
  python3 exp/run_eval.py --kb "$1" --items "$2" --nli-model "$SPEC" --retrieval "$3" --out "$4.part"
  mv "$4.part" "$4"
}

# 1. course (KCA)
[[ -s exp/data/circular__kca.json ]] || python3 exp/make_circular.py --kb exp/kbs/kca.jsonl --out exp/data/circular__kca.json
run exp/kbs/kca.jsonl exp/data/errors_noncircular.json    concept exp/runs/kca__${TAG}__concept__errors.jsonl
run exp/kbs/kca.jsonl exp/data/true_statements_DRAFT.json concept exp/runs/kca__${TAG}__concept__true.jsonl
run exp/kbs/kca.jsonl exp/data/circular__kca.json         concept exp/runs/kca__${TAG}__concept__circular.jsonl

# 2. ProgMiscon (KCA, open retrieval)
mkdir -p exp/runs_progmiscon
run exp/kbs/kca.jsonl exp/data/progmiscon_errors.json open exp/runs_progmiscon/kca__${TAG}__open__errors.jsonl
run exp/kbs/kca.jsonl exp/data/progmiscon_true.json   open exp/runs_progmiscon/kca__${TAG}__open__true.jsonl

# 3. ProgMiscon oracle KB
[[ -s exp/kbs/progmiscon_oracle.jsonl ]] || python3 exp/make_progmiscon_oracle.py
mkdir -p exp/runs_progmiscon_oracle
run exp/kbs/progmiscon_oracle.jsonl exp/data/progmiscon_errors.json          open exp/runs_progmiscon_oracle/progmiscon_oracle__${TAG}__open__errors.jsonl
run exp/kbs/progmiscon_oracle.jsonl exp/data/progmiscon_oracle_template.json open exp/runs_progmiscon_oracle/progmiscon_oracle__${TAG}__open__template.jsonl

echo "Done. Now:"
echo "  python3 exp/analyze.py --accepted-only --out exp/results/accepted 2>&1 | grep kca"
echo "  python3 exp/analyze_progmiscon.py"
