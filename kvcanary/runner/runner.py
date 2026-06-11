import json
import os
import time

from kvcanary.types import CellResult


def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue   # truncated tail line from a crash; it'll be re-written
                keys.add((r["cell_id"], r["sample_id"]))
    return keys


def run_cells(cells, backend, compressor_factory, tasks, out_path: str) -> int:
    done = _done_keys(out_path)
    written = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        for spec in cells:
            comp = compressor_factory(spec.method, spec.budget)
            task = tasks[spec.task]
            for sample in task.build_samples():
                if (spec.cell_id, sample.id) in done:
                    continue
                t0 = time.perf_counter()
                if task.needs_generation:
                    gen = backend.generate(sample.prompt, compressor=comp)
                    res = task.evaluate(sample, output=gen.text)
                    raw, n_tok, kv = gen.text, gen.n_tokens, gen.kv_bytes_retained
                else:
                    ppl = backend.score(sample.prompt, compressor=comp)
                    res = task.evaluate(sample, output="", ppl=ppl)
                    full_bytes = getattr(backend, "full_bytes", 1000.0)
                    raw, n_tok, kv = "", 0, comp.kv_bytes_retained(full_bytes)
                row = CellResult(
                    spec=spec,
                    sample_id=sample.id,
                    scores=res.scores,
                    raw_output=raw,
                    n_tokens=n_tok,
                    latency_s=time.perf_counter() - t0,
                    kv_bytes_retained=kv,
                )
                f.write(json.dumps(row.to_row()) + "\n")
                f.flush()
                written += 1
    return written
