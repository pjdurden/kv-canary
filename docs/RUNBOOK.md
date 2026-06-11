# kv-canary — GPU run runbook

The CPU smoke (`configs/smoke.yaml`, distilgpt2) proves the pipeline end-to-end. The real finding
needs a GPU. Steps:

1. **Rent a 24GB spot GPU** (4090 / A10 / L4). `pip install -e ".[dev,ml]"`.
2. **Wire real eviction into HF's KV cache.** This is the one deferred piece. `HFBackend` currently
   records `kv_bytes_retained` but does NOT yet apply eviction/quantization during the forward pass
   (see the class docstring in `kvcanary/backends/hf.py`): until this lands, ALL compressors produce
   identical text/perplexity and the divergence will be NULL. Implement a custom `DynamicCache`
   subclass whose `update()` calls `compressor.kept_indices(...)` (the pure selection logic is
   already tested in Tasks 4–6) and pass it into `model.generate(..., past_key_values=...)`.
   Validate on `Qwen2.5-0.5B` first.
3. **Run the matrix** (resumable — safe to re-run after a spot kill):
   `python -m kvcanary run configs/v1.yaml --out results/raw/v1.jsonl`
4. **Aggregate + report:** `python scripts/aggregate_and_report.py results/raw/v1.jsonl`
5. **Inspect `report/divergence.png`:** the eviction methods' functional (solid) lines should cliff
   while their perplexity-quality (dashed) lines hug 1.0 — that gap is the finding. The SDS column in
   `report/RESULTS.md` quantifies it per (method, budget).
6. **CONTINUE** if the divergence reproduces on >=1 eviction method + task (independently confirming
   the VeriCache effect). **KILL / pivot** to the benchmark-coverage harness if functional
   degradation tracks perplexity 1:1 (no divergence) — see spec §8.

## Caveats (carry into the write-up)
- The wedge is functional-vs-token-level DIVERGENCE, not "reproducible benchmarking" (that gap is
  coverage, not reproducibility — see spec §9).
- Quant and eviction share a memory-retained x-axis but are different mechanisms; the differing
  degradation is the point.
