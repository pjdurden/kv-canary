# kv-canary — cloud-GPU run runbook

The CPU smoke (`configs/smoke.yaml`, distilgpt2) proves the pipeline end-to-end; the real finding
needs a GPU on a real instruct model. Compression is **wired** (`CompressedCache` + `HFBackend`) —
this is now a *run*, not a build.

## TL;DR

```bash
# on a 24GB spot GPU (4090 / A10 / L4):
git clone <repo> && cd kv-canary
./scripts/run_gpu.sh validate     # preflight + Qwen2.5-0.5B gate matrix + report (cheap, ~minutes)
#   ...inspect report/divergence.png + report/RESULTS.md...
./scripts/run_gpu.sh full         # Qwen2.5-7B matrix + report (the run)
```

Both stages are **resumable** — re-running skips completed cells in `results/raw/*.jsonl`, so a spot
kill costs only the in-flight cell. Tune with env: `DEVICE DATA LIMIT MAXNEW DTYPE`.

## What each stage does

1. **`validate`** — runs `scripts/probe_rope.py` (a 3-method sanity check on Qwen2.5-0.5B that
   catches wiring breakage in ~30s) then the `configs/v1-validate.yaml` matrix and report. This is
   the **GATE**: only proceed if eviction methods' functional (solid) lines cliff while their
   perplexity-quality (dashed) lines hug 1.0. If functional degradation tracks perplexity 1:1
   (no divergence), KILL / pivot to the benchmark-coverage harness (spec §8).
2. **`full`** — `configs/v1.yaml` (Qwen2.5-7B, full budget sweep) + report.

## Efficiency / cost notes

- **dtype**: `bf16` on CUDA by default (Qwen2.5's native dtype; fp16 can overflow). 7B in bf16 ≈ 14GB
  weights — fits a 24GB card with room for the (short) activations.
- **attention**: forced `eager` — REQUIRED because SnapKV reads prefill attention via
  `output_attentions`, which sdpa/flash return empty (verified on transformers 5.x). At these short
  prompt lengths (HumanEval ≲150 tok, tool ≲50, ppl context ≲ half a paragraph) eager's overhead
  over sdpa is negligible, so it is not worth switching impl per cell.
- **`--limit`** caps samples/task and **`--max-new-tokens`** caps generation — both lower cost. The
  full matrix is 12 compressors × 3 tasks × LIMIT samples; budget GPU-hours accordingly.
- Greedy decode for lossy methods runs a Python loop (one forward/token) — fine for short outputs.

## Known fidelity caveat — carry into the write-up

- **RoPE re-positioning.** HF caches keys post-RoPE; after eviction the kept keys keep their original
  rotary phase while the next decode token is placed at the post-eviction cache length, so relative
  positions are slightly distorted (true StreamingLLM re-ropes the kept keys; this does not).
  Empirically this does **not** break the run: on Qwen2.5-0.5B (CPU) output stays coherent and
  divergence is large (full ppl ≈ 5 → streamingllm@0.25 ≈ 20 → snapkv@0.25 ≈ 364). Absolute numbers
  could shift under proper re-roping, so report the method as "eviction over HF post-RoPE cache" and
  flag re-roping as the obvious refinement.
- The wedge is functional-vs-token-level **divergence**, not "reproducible benchmarking" (that gap is
  coverage, not reproducibility — spec §9).
- Quant and eviction share a memory-retained x-axis but are different mechanisms; the differing
  degradation at equal budget is the point.
- Tool-calls use kv-canary's bundled FC set (BFCL/xLAM are non-HF-native / gated); code = HumanEval,
  perplexity = WikiText-2 (both real, via `--data real`).
