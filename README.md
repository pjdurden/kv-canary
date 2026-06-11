# kv-canary

**A canary for *silent* KV-cache-compression failures.**

Lossy KV-cache compression (quantization, token eviction) is benchmarked almost entirely on
**speed and memory** — and on token-level quality (perplexity). But token-level metrics can stay
flat while **functional** outputs silently break: generated code stops passing its tests, tool
calls stop being valid JSON. kv-canary measures that gap.

> The wedge: everyone measures how *fast* KV compression is. Almost nobody measures whether it
> silently breaks your code and tool calls while perplexity says everything is fine.

This is a measurement harness + methodology, motivated by VeriCache
([arXiv:2605.17613](https://arxiv.org/html/2605.17613)), which showed lossy-KV bias accumulates
linearly and functional tasks fail while token-level metrics look unchanged. kv-canary makes that
effect reproducible and quantifiable across compression methods.

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

**v1 is the harness.** 16 build steps, ~38 tests, green CI. The real run is documented in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) and needs a single 24GB GPU.

- **The headline finding is not in this repo yet.** One piece is intentionally deferred to the GPU
  run: wiring the eviction selection into HuggingFace's live KV cache (a custom `DynamicCache`
  subclass). Until that lands, `HFBackend` records *how much* KV each method would retain but runs
  normal generation — so **all methods produce identical output and the divergence is NULL.** This
  is called out in the `HFBackend` docstring and the runbook. Any numbers you see in a freshly
  generated `report/` from the CPU smoke are pipeline-shape placeholders, **not** a result.
- The gap kv-canary targets rests on a single recent paper (VeriCache); the contribution here is
  making it *measurable and reproducible*, not claiming novelty of the phenomenon.
- The wedge is **functional-vs-token-level divergence**, not "reproducible benchmarking" — local
  inference benchmarking is fragmented (a coverage gap), but that is a different problem.

## Layout

```
kvcanary/        compressors/ · backends/ · tasks/ · runner/ · metrics/ · report/
configs/         smoke.yaml (CPU) · v1.yaml (GPU matrix)
scripts/         aggregate_and_report.py
docs/            RUNBOOK.md · superpowers/specs/ (design) · superpowers/plans/ (build plan)
```
