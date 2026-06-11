import json
import subprocess
import sys

import pytest

pytest.importorskip("transformers")


@pytest.mark.slow
def test_cli_smoke_end_to_end(tmp_path):
    out = tmp_path / "raw.jsonl"
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(
        "models: [distilgpt2]\n"
        "tasks: [tool, perplexity]\n"
        "compressors:\n"
        "  - {method: full, budget: 1.0}\n"
        "  - {method: streamingllm, budget: 0.5}\n"
    )
    r = subprocess.run([sys.executable, "-m", "kvcanary", "run", str(cfg),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) >= 2 and any(row["task"] == "perplexity" for row in rows)
    assert all("scores" in row and "latency_s" in row for row in rows)
