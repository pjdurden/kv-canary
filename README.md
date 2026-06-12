# kv-canary

[![CI](https://github.com/pjdurden/kv-canary/actions/workflows/ci.yml/badge.svg)](https://github.com/pjdurden/kv-canary/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

**A canary for *silent* KV-cache-compression failures.**

Lossy KV-cache compression (quantization, token eviction) silently breaks **functional** outputs:
generated code stops passing its tests, tool calls stop being valid JSON. VeriCache
([arXiv:2605.17613](https://arxiv.org/abs/2605.17613)) shows these methods are "inherently
lossy … their outputs increasingly diverge from full-KV-cache outputs as more tokens are decoded,
which leads to catastrophic failures in code generation and tool calling." *The Pitfalls of KV Cache
Compression* ([arXiv:2510.00231](https://arxiv.org/abs/2510.00231), ACL 2026) shows aggregate
benchmark metrics hide these per-instruction failures.

> The question kv-canary tests: when KV compression breaks your code and tool calls, does the cheap
> token-level metric (perplexity) **warn you, or stay silent?** This is a *hypothesis under test* —
> no paper has shown perplexity is blind to KV-compression damage; kv-canary measures whether it is.

**Scope & prior art (honest).** Functional/downstream degradation under KV compression is *already*
measured by [NVIDIA KVPress](https://github.com/NVIDIA/kvpress), Pitfalls (IFEval), and
[arXiv:2512.12008](https://arxiv.org/abs/2512.12008) (8 reasoning benchmarks). What those don't score
is **executable code (pass@k) and JSON/tool-call schema conformance** specifically — that narrow
target, contrasted against perplexity, is kv-canary's niche. See
[`docs/RESEARCH.md`](docs/RESEARCH.md) for the full cited gap audit, including known implementation
limitations (StreamingLLM re-roping, per-head SnapKV) that a faithful run must address.

---

## The idea in one chart

For each compression method, kv-canary plots two lines against **KV memory retained**:

- **functional accuracy** (solid) — does generated code pass its tests / is the tool call correct?
- **perplexity-implied quality** (dashed) — the token-level metric, normalized to the full cache.

The finding is the **divergence**: functional accuracy cliffing while the perplexity line hugs 1.0.
That visual gap is the whole point — the token metric tells you nothing is wrong while real
functional quality collapses.

To make it a single number, kv-canary defines the **Silent Degradation Score (SDS)**:

```
SDS = (relative functional drop) / (relative perplexity rise)
```

High SDS = functional accuracy craters while perplexity barely moves (the metric "lies"). SDS ≈ 1
is graceful degradation (perplexity warns you in step). Perplexity *improving* while function
breaks is the most deceptive case of all, and SDS reports it as maximally silent — by design.

## What it measures

| Compression family | Methods | How `budget` maps |
|---|---|---|
| baseline | full (fp16) | 1.0 |
| quantization | int8 / int4 / int2 KV | bits / 16 |
| token eviction | StreamingLLM, SnapKV | fraction of tokens kept |

Against two objective, KV-eviction-sensitive functional tasks — **code execution** (pass@1) and
**tool/JSON calling** (valid + correct-function + arg-match) — contrasted with **perplexity** on a
held-out set. All on a shared *KV-memory-retained* x-axis, so quantization and eviction are directly
comparable (and the fact that equal memory budgets degrade them differently is the finding).

## Quickstart (CPU, no GPU)

Verifies the whole pipeline end-to-end on a tiny model:

```bash
pip install -e ".[dev,ml]"
python -m kvcanary run configs/smoke.yaml --out results/raw/smoke.jsonl
python scripts/aggregate_and_report.py results/raw/smoke.jsonl
# -> report/divergence.png, report/RESULTS.md
```

> The smoke run uses `distilgpt2` and proves the harness runs. See the note on the v1 deferral below
> for why the *finding* itself needs a GPU.

Run the fast test suite:

```bash
pip install -e ".[dev]"
pytest -q          # model-free unit tests
pytest -q -m slow  # + end-to-end smoke on distilgpt2 (downloads weights)
```

## How it works

Three small, independently-tested seams compose into a resumable experiment runner:

- **`KVCompressor`** — `full`, `quantized`, `streamingllm`, `snapkv` behind one `budget` knob; the
  eviction selection logic is a pure function of attention scores (unit-tested on synthetic tensors).
- **`Backend`** — `HFBackend` (HuggingFace causal LM: greedy generation + teacher-forced
  perplexity) and `FakeBackend` (a deterministic double, so the runner and tasks test with no model).
- **`Task`** — `CodeExecTask` (sandboxed subprocess, pass@1), `ToolCallTask` (JSON parse + function
  + arg match), `PerplexityTask`.

The runner sweeps `models × compressors × tasks`, writes one JSON line per sample, and **resumes**
by skipping already-completed cells — so a killed spot-GPU run restarts cheaply. `metrics/` and
`report/` turn the JSONL into the divergence chart and the SDS table.

```
configs/*.yaml ─▶ runner ─▶ results/raw/*.jsonl ─▶ aggregate ─▶ report/{divergence.png, RESULTS.md}
                    │
        KVCompressor · Backend · Task
```

## Status & honest caveats

**v1 is the harness, wired end-to-end.** Compression is applied into HF's live KV cache
(`CompressedCache`), `score()` measures perplexity under the compressed context, and the divergence
is real (not NULL). The headline GPU run on a real instruct model is documented in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md); run it before trusting any numbers. Full cited audit in
[`docs/RESEARCH.md`](docs/RESEARCH.md). Known limitations a faithful run must weigh:

- **The method implementations are simplified and a faithful reviewer would flag two of them.**
  StreamingLLM here evicts from a *post-RoPE* HF cache **without re-roping** kept keys to contiguous
  cache positions — the real method caches keys pre-RoPE and re-indexes positions, so eviction here
  is degraded partly *for the wrong reason* (see RESEARCH.md Q3). SnapKV here **averages attention
  over heads** with no pooling, whereas real SnapKV selects **per-head with a max-pool clustering
  step** (Q4). Treat these as "StreamingLLM-/SnapKV-style" eviction, not exact reproductions.
- **"Perplexity is blind" is a hypothesis under test, not a cited result.** VeriCache shows the
  functional failures; no paper shows perplexity fails to warn. kv-canary measures whether it does.
- **Prior art exists.** KVPress / Pitfalls / arXiv:2512.12008 already measure functional/downstream
  degradation under KV compression; kv-canary's narrow niche is executable-code + tool-schema
  conformance specifically. The originality is that corner, not the phenomenon.
- **Design scope.** Compression is applied to the prompt once (no decode-time rolling); VeriCache's
  failures accrue over long *decode*, so this design under-exercises that regime (RESEARCH.md §Deeper
  problems). Quant is per-tensor absmax and `bits/16` ignores scale/zero-point overhead.

## Layout

```
kvcanary/        compressors/ · backends/ · tasks/ · runner/ · metrics/ · report/
configs/         smoke.yaml (CPU) · v1.yaml (GPU matrix)
scripts/         aggregate_and_report.py
docs/            RUNBOOK.md · superpowers/specs/ (design) · superpowers/plans/ (build plan)
```

## License

MIT — see [LICENSE](LICENSE).
