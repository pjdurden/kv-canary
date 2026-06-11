import json
from kvcanary.types import CellSpec
from kvcanary.backends.fake import FakeBackend
from kvcanary.compressors.full import FullCache
from kvcanary.compressors.streamingllm import StreamingLLMCompressor
from kvcanary.tasks.tool_call import ToolCallTask
from kvcanary.tasks.perplexity import PerplexityTask
from kvcanary.runner.runner import run_cells

def _factory(method, budget):
    if method == "full":
        return FullCache(budget)
    return StreamingLLMCompressor(budget)

def _setup():
    prob = {"id": "w", "prompt": "p", "gold": {"name": "f", "arguments": {}}}
    backend = FakeBackend(scripted={"p": '{"name": "f", "arguments": {}}'})
    tasks = {"tool": ToolCallTask([prob])}
    cells = [CellSpec("toy", "full", 1.0, "tool"),
             CellSpec("toy", "streamingllm", 0.5, "tool")]
    return backend, tasks, cells

def test_runner_writes_rows_and_resumes(tmp_path):
    out = tmp_path / "raw.jsonl"
    backend, tasks, cells = _setup()
    n1 = run_cells(cells, backend, _factory, tasks, str(out))
    assert n1 == 2                                   # 2 cells x 1 sample
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["cell_id"] for r in rows} == {"toy|full|1.0|tool", "toy|streamingllm|0.5|tool"}
    assert all(r["raw_output"] == '{"name": "f", "arguments": {}}' for r in rows)  # generated text stored
    n2 = run_cells(cells, backend, _factory, tasks, str(out))   # resume
    assert n2 == 0                                   # everything already done
    assert len(out.read_text().splitlines()) == 2    # no duplicate rows


def test_runner_perplexity_path_records_score_and_bytes(tmp_path):
    out = tmp_path / "ppl.jsonl"
    backend = FakeBackend(scripted={}, ppl=42.0, full_bytes=2000.0)
    tasks = {"perplexity": PerplexityTask(texts=["hello world"])}
    cells = [CellSpec("toy", "streamingllm", 0.5, "perplexity")]
    n = run_cells(cells, backend, _factory, tasks, str(out))
    assert n == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert row["scores"]["perplexity"] == 42.0
    assert row["n_tokens"] == 0
    assert row["kv_bytes_retained"] == 1000.0          # 0.5 * full_bytes(2000)

def test_runner_partial_resume_writes_only_missing(tmp_path):
    out = tmp_path / "raw.jsonl"
    prob1 = {"id": "a", "prompt": "p", "gold": {"name": "f", "arguments": {}}}
    prob2 = {"id": "b", "prompt": "p", "gold": {"name": "f", "arguments": {}}}
    backend = FakeBackend(scripted={"p": '{"name": "f", "arguments": {}}'})
    tasks = {"tool": ToolCallTask([prob1, prob2])}
    cells = [CellSpec("toy", "full", 1.0, "tool")]
    # pre-seed one (cell_id, sample_id) row as already done
    out.write_text(json.dumps({"cell_id": "toy|full|1.0|tool", "sample_id": "a"}) + "\n")
    n = run_cells(cells, backend, _factory, tasks, str(out))
    assert n == 1                                       # only sample "b" was missing
    ids = {json.loads(line)["sample_id"] for line in out.read_text().splitlines()}
    assert ids == {"a", "b"}                            # no duplicate of "a"
