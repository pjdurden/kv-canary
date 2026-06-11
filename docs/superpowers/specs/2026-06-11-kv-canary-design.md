# kv-canary — Design Spec

_Date: 2026-06-11. A canary for **silent** KV-cache-compression failures._

## 1. Purpose & success criteria

**Problem.** Lossy KV-cache compression (quantization, token eviction) is benchmarked almost
entirely on **speed/memory** and on **token-level quality** (perplexity). But token-level metrics
can stay flat while **functional** outputs silently break — generated code stops passing tests, tool
calls stop being valid/correct JSON. VeriCache (arXiv 2605.17613, May 2026) flagged that lossy-KV
bias accumulates linearly and functional tasks fail while token metrics look fine. **No polished
open-source tool measures this.**

**What kv-canary is.** A harness that, for a given model and compression method, measures the gap
between **functional task accuracy** and **token-level quality** as a function of how much KV memory
is retained — surfacing the "silent" failure region.

**Primary artifact (finding-first).** One compelling, reproducible result:
> "Method X at compression ratio R: perplexity barely moves, but functional accuracy collapses."

packaged as a divergence chart + `RESULTS.md` + a runnable repo. The reusable tool (clean
interfaces below) emerges from the harness built to produce that finding.

**Success criteria (v1):**
- Reproduce the divergence (functional accuracy cliff while perplexity stays ~flat) on at least one
  eviction method × one task, with raw per-sample data.
- A single chart + a short write-up that a stranger can re-run with one command.
- The harness itself is tested (a correctness tool must be correct).

**Non-goals (v1):** not a production serving engine; not patching vLLM/SGLang; not exhaustive method
coverage; not a research paper (write-up is engineering-blog grade).

**Audience / why it fits the builder.** Portfolio artifact for a remote AI-infra / inference-serving
role. Same genre as the builder's recent OSS PRs (candle shape-validation, gguf overflow, tokenizers
overflow): correctness-under-adversarial-conditions where the naive metric looks fine but output is
silently wrong. Unifies them into one brand: **correctness in inference systems.**

## 2. Constraints

- Solo, ~5–10 hrs/week.
- **Compute:** develop on laptop (CPU, toy model, $0); run real experiments on a **rented single
  spot GPU** (24GB, e.g. 4090/A10/L4). Full v1 matrix ≈ a few GPU-hours ≈ $10–30.
- Stack: Python + HuggingFace `transformers` (full CUDA ecosystem on the rented GPU).
- No crypto/blockchain angle. Pure AI inference infra.

## 3. Architecture & components

Python package `kvcanary`. Three clean seams (the reusable-tool surface) + orchestration.

### Seam 1 — `compressors/` : `KVCompressor`
ABC with one uniform knob, `budget`. Implementations:
- `FullCache` — baseline fp16 (must be a no-op).
- `QuantizedCache` — wraps HF built-in quantized KV (int8/int4/int2).
- `SnapKVCompressor`, `StreamingLLMCompressor` — eviction; `budget` = fraction of tokens kept.
- `H2OCompressor` — optional, post-v1.

All methods sit behind one API so the runner sweeps them identically.

### Seam 2 — `backends/` : `Backend`
- Interface: `generate(prompt, compressor, **kw) -> text`; `score(text, compressor) -> perplexity`.
- `HFBackend` — loads a HF causal LM, installs the compressor's cache, runs greedy generation +
  teacher-forced perplexity. **Only backend in v1.** The interface is the explicit seam where a
  future `VLLMBackend` would plug in (infra-thinking signal, no vLLM patching now).

### Seam 3 — `tasks/` : `Task`
- Interface: `build_samples() -> list[Sample]`; `evaluate(sample, output) -> Score`.
- `CodeExecTask` — HumanEval/MBPP subset; runs generated code against unit tests in a
  **subprocess sandbox** (timeout, no network) → pass@1. A long preamble is prepended so eviction
  bites otherwise-short prompts.
- `ToolCallTask` — tool schema + conversation → model emits a call; `evaluate` reports
  JSON-parseable, correct-function, and arg-match (three sub-scores). Dataset: a BFCL subset or a
  hand-built set.
- `PerplexityTask` — held-out long-context text → perplexity (the token-level metric that "looks
  fine").

### Orchestration
- `runner/` — reads YAML config, expands the (model × compressor × budget × task) matrix, runs it,
  writes per-sample JSONL + aggregates. **Idempotent/resumable**: skips cells whose rows already
  exist, so a killed spot instance resumes cheaply.
- `metrics/` — aggregates per (method, budget); computes the headline divergence (§5).
- `report/` — renders the chart + `RESULTS.md`.
- `configs/` (YAML) + CLI: `python -m kvcanary run configs/v1.yaml`.

**Isolation:** compressors don't know tasks; tasks don't know compressors; backend bridges; runner
orchestrates; metrics/report consume. Each unit is independently testable. The three seams are the
OSS-tool API.

## 4. Data flow

Single cell = (model, compressor, budget, task, sample).
1. `runner` expands YAML → cells.
2. Per (model, compressor, budget): `HFBackend` loads the model once, installs the compressor at
   that budget.
3. Per task × sample: build prompt → `backend.generate()` / `.score()` → `task.evaluate()` →
   append row to `results/raw/*.jsonl` (model, method, budget, task, sample_id, score, raw_output,
   tokens, latency, kv_bytes_retained).
4. Resume: skip a cell whose rows exist.
5. After matrix: `metrics` → `results/agg/*.csv`; `report` → chart + `RESULTS.md`.

**Shared x-axis:** compression ratio = **fraction of full KV memory retained** (quant int4 = 0.25,
int2 = 0.125; eviction keeping 50% = 0.5). Physically comparable (bytes saved) across families — and
the fact that *equal memory budget* degrades functional accuracy very differently for quant vs
eviction is the finding.

## 5. Metrics & the headline finding

Per (method, budget), aggregate:
- **Functional:** pass@1 (code); tool-call JSON-valid rate, correct-function rate, arg-match rate.
- **Token-level:** perplexity on held-out long-context text. (Optional: greedy-output token overlap
  vs FullCache — "the text looks the same" while functionally broken.)
- **Confirmation:** measured KV bytes retained (validates the x-axis); latency.

**Headline coined metric — Silent Degradation Score (SDS).** At memory budget r:

    SDS(method, r) = (relative functional drop) / max(relative perplexity drop, ε)
                   = (Δfunctional / functional_full) / max(Δppl / ppl_full, ε)

High SDS = functional accuracy craters while perplexity barely moves — the token metric "lies." The
ε floor on the denominator prevents blow-up when Δppl ≈ 0; SDS is a derived *silentness indicator*,
**not** the primary claim.

**Primary artifact = the divergence chart + the raw drop table.** Chart: x = KV memory retained
(1.0 → 0.1, log); y = metric normalized to FullCache (1.0 = no degradation); per method a color;
functional accuracy (solid) vs perplexity-implied quality (dashed). The visual: perplexity lines
hug 1.0 while eviction functional lines fall off a cliff. The table reports
(Δfunctional%, Δppl%, SDS) per (method, budget). Example headline form:
> "SnapKV at 25% KV memory: perplexity +3%, tool-call accuracy −41% → SDS ≈ 14×."

## 6. Testing strategy (the correctness tool must be correct)

This section doubles as portfolio signal — rigorously tested harness, rare for research code.
- **Compressor invariants:** `FullCache` is a true no-op (logits identical to no-compression);
  eviction at budget=1.0 (keep all) == `FullCache`; quant at full precision ≈ `FullCache` within
  tolerance.
- **Task evaluators:** `CodeExecTask` passes a known-correct solution, fails a known-wrong one;
  sandbox enforces timeout + no network. `ToolCallTask`: golden valid call passes; malformed JSON,
  wrong function, wrong arg each fail the right sub-score.
- **Backend:** greedy generation deterministic under fixed seed; perplexity matches a hand-computed
  value on a tiny input.
- **Runner:** resume skips completed cells; matrix expansion is correct.
- **CPU smoke test in CI (GitHub Actions):** full pipeline on a toy model (Qwen2.5-0.5B) + 3 samples
  runs green — this is also the reproducibility proof for reviewers.

## 7. Experiment matrix (v1)

- **Models:** Qwen2.5-7B-Instruct (strong code + tool-calling, long context, one 24GB GPU). Add
  Llama-3.1-8B-Instruct *if time* (shows the finding isn't model-specific).
- **Compressor × budget per model:** FullCache (1) + Quantized {int8, int4, int2} (3) +
  SnapKV {75/50/25/12.5% kept} (4) + StreamingLLM {same 4} (4) ≈ **12 cells**.
- **Tasks:** CodeExec (100-problem subset, long preamble) + ToolCall (100 samples) + Perplexity
  (a few long docs).
- **Volume:** ~12 cells × ~200 functional samples ≈ 2,400 generations + perplexity → a few
  GPU-hours. Two models ≈ double.

## 8. Milestones (the 2-week wedge) & distribution

- **Week 1:** seams + `HFBackend` + `FullCache` + `SnapKV` + `CodeExecTask` + `PerplexityTask`,
  running end-to-end on CPU toy model (green CI). First real run on rented GPU with Qwen2.5-7B →
  first divergence data point.
- **Week 2:** add `QuantizedCache` + `StreamingLLM` + `ToolCallTask`; full budget sweep; metrics +
  chart + `RESULTS.md`; write the blog post / README around the money chart.
- **Shareable finding:** README + short write-up ("Your KV-cache compression is silently breaking
  your tool calls"), the divergence chart, the SDS table, reproducible via one command. Distribution:
  the builder's existing audience + r/LocalLLaMA + HN + the relevant method repos' discussions.

**Continue / kill (from the research wedge):**
- **Continue** if SnapKV/StreamingLLM show a functional cliff while perplexity stays ~flat on ≥1
  task — independently reproducing the VeriCache effect.
- **Kill / pivot** to the benchmark-coverage harness if functional degradation tracks perplexity
  1:1 across methods (no divergence).

## 9. Honest caveats (carry into the write-up)

- The gap rests on a single recent paper (VeriCache); kv-canary's value is making it
  measurable/reproducible, not claiming novelty of the phenomenon.
- "No primary hiring evidence" was found in research — the credibility argument is first-principles,
  not data. The artifact must stand on its own engineering merit.
- The benchmark gap in this space is **coverage/fragmentation, not reproducibility** — do not pitch
  "reproducible benchmarking" as the wedge; the wedge is **functional vs token-level divergence**.
- Quant "ratio" and eviction "kept-fraction" share a memory-retained axis but are not physically
  identical mechanisms — present them as such; the differing degradation is the point.
