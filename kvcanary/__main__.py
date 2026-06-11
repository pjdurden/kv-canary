import argparse
import json

from kvcanary.runner.config import load_cells
from kvcanary.runner.registry import build_compressor
from kvcanary.backends.hf import HFBackend
from kvcanary.tasks.tool_call import ToolCallTask
from kvcanary.tasks.code_exec import CodeExecTask
from kvcanary.tasks.perplexity import PerplexityTask
from kvcanary.runner.runner import run_cells

# v1 ships small built-in sample sets; swap for HumanEval/BFCL subsets on the GPU run.
_TOOL = [{"id": "weather", "prompt": "Return JSON calling get_weather for Paris.",
          "gold": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
_CODE = [{"id": "add", "prompt": "Write a python function add(a,b) returning a+b. Only code.",
          "check": "assert add(2,3)==5\n"}]
_PPL = ["The quick brown fox jumps over the lazy dog near the river bank at dawn."]


def main():
    ap = argparse.ArgumentParser(prog="kvcanary")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("config")
    runp.add_argument("--out", default="results/raw/run.jsonl")
    args = ap.parse_args()

    cells = load_cells(args.config)
    models = sorted({c.model for c in cells})
    if len(models) > 1:
        raise ValueError(
            f"v1 runs one model per config (the runner uses a single backend for all cells); "
            f"got {models}. Split into separate config files / runs."
        )
    backend = HFBackend(model_id=models[0], device="cpu")
    tasks = {"tool": ToolCallTask(_TOOL), "code": CodeExecTask(_CODE),
             "perplexity": PerplexityTask(_PPL)}
    n = run_cells(cells, backend, build_compressor, tasks, args.out)  # factory, per-cell
    print(json.dumps({"written": n, "out": args.out}))


if __name__ == "__main__":
    main()
