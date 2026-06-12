#!/usr/bin/env bash
# One-shot kv-canary cloud-GPU pipeline. Resumable: re-running skips completed cells
# (results/raw/*.jsonl), so a spot-instance kill costs only the in-flight cell.
#
#   ./scripts/run_gpu.sh validate   # preflight + Qwen2.5-0.5B gate matrix + report  (cheap)
#   ./scripts/run_gpu.sh full       # Qwen2.5-7B matrix + report                     (the run)
#
# Override via env: DEVICE=cuda DATA=real LIMIT=20 MAXNEW=256 DTYPE=bf16
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-validate}"
DEVICE="${DEVICE:-cuda}"; DATA="${DATA:-real}"; LIMIT="${LIMIT:-20}"
MAXNEW="${MAXNEW:-256}"; DTYPE="${DTYPE:-bf16}"
COMMON=(--device "$DEVICE" --data "$DATA" --limit "$LIMIT" --max-new-tokens "$MAXNEW" --dtype "$DTYPE")

echo "==> install deps"
pip install -q -e ".[ml,dev]"

if [[ "$STAGE" == "validate" ]]; then
  echo "==> preflight: RoPE sanity on Qwen2.5-0.5B (catches wiring breakage before the 7B spend)"
  python scripts/probe_rope.py

  echo "==> validate matrix (Qwen2.5-0.5B)"
  python -m kvcanary run configs/v1-validate.yaml --out results/raw/validate.jsonl "${COMMON[@]}"
  python scripts/aggregate_and_report.py results/raw/validate.jsonl
  echo
  echo "GATE: inspect report/divergence.png + report/RESULTS.md."
  echo "  CONTINUE -> ./scripts/run_gpu.sh full   (if eviction functional lines cliff while ppl-quality hugs 1.0)"
  echo "  KILL/PIVOT                              (if functional degradation tracks perplexity 1:1)"
elif [[ "$STAGE" == "full" ]]; then
  echo "==> full matrix (Qwen2.5-7B) — resumable"
  python -m kvcanary run configs/v1.yaml --out results/raw/v1.jsonl "${COMMON[@]}"
  python scripts/aggregate_and_report.py results/raw/v1.jsonl
  echo "DONE -> report/divergence.png, report/RESULTS.md"
else
  echo "usage: $0 [validate|full]"; exit 2
fi
