import argparse
import json

from kvcanary.runner.config import load_cells
from kvcanary.runner.registry import build_compressor
from kvcanary.backends.hf import HFBackend
from kvcanary.tasks.tool_call import ToolCallTask
from kvcanary.tasks.code_exec import CodeExecTask
from kvcanary.tasks.perplexity import PerplexityTask
from kvcanary.tasks.loaders import load_code_problems, load_ppl_texts, load_tool_problems
from kvcanary.runner.runner import run_cells


def main():
    ap = argparse.ArgumentParser(prog="kvcanary")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("config")
    runp.add_argument("--out", default="results/raw/run.jsonl")
    runp.add_argument("--device", default="cpu",
                      help="torch device, e.g. cpu or cuda (use cuda for the v1 GPU run)")
    runp.add_argument("--data", choices=["builtin", "real"], default="builtin",
                      help="builtin = bundled offline sets (CI/smoke); real = HumanEval/WikiText "
                           "via `datasets` (network). Tool-calls are always the bundled FC set.")
    runp.add_argument("--limit", type=int, default=None,
                      help="max samples per task (default: all available in the chosen source)")
    args = ap.parse_args()

    cells = load_cells(args.config)
    models = sorted({c.model for c in cells})
    if len(models) > 1:
        raise ValueError(
            f"v1 runs one model per config (the runner uses a single backend for all cells); "
            f"got {models}. Split into separate config files / runs."
        )
    backend = HFBackend(model_id=models[0], device=args.device)
    n = args.limit if args.limit is not None else 10**9   # None -> take everything available
    tasks = {
        "tool": ToolCallTask(load_tool_problems(n, source=args.data)),
        "code": CodeExecTask(load_code_problems(n, source=args.data)),
        "perplexity": PerplexityTask(load_ppl_texts(n, source=args.data)),
    }
    written = run_cells(cells, backend, build_compressor, tasks, args.out)  # factory, per-cell
    print(json.dumps({"written": written, "out": args.out,
                      "device": args.device, "data": args.data}))


if __name__ == "__main__":
    main()
