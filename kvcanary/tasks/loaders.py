"""Task-subset loaders: real benchmark data for the GPU run, offline fallbacks for CI/smoke.

``source="real"`` pulls canonical benchmarks via ``datasets`` (network, ``[ml]`` extra):
  * code        -> HumanEval (``openai/openai_humaneval``)
  * perplexity  -> WikiText-2 raw (``Salesforce/wikitext``)
``source="builtin"`` (the default) uses small bundled sets so the package imports, CI runs, and
the CPU smoke work with no network and no ``datasets`` install.

Tool-calling is the exception: the canonical FC benchmarks are awkward to consume — BFCL is not
stored as an HF-native dataset, xLAM is gated, and ToolACE encodes calls in a bespoke non-JSON DSL.
Rather than ship a brittle parser and call it "real," kv-canary bundles its own curated single-call
set (``kvcanary/data/tool_calls.jsonl``) matching the ToolCallTask ``{name, arguments}`` gold schema.
``load_tool_problems`` therefore ignores ``source``; swap in a BFCL/xLAM adapter here if you have one.
"""
import json
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")

# --- offline fallbacks: complete-function prompts (model emits the whole def) + checks ---
_BUILTIN_CODE = [
    {"id": "add", "prompt": "Write a complete python function add(a, b) that returns a+b. Only code.",
     "check": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"},
    {"id": "is_palindrome",
     "prompt": "Write a complete python function is_palindrome(s) returning True iff s reads the "
               "same forwards and backwards. Only code.",
     "check": "assert is_palindrome('racecar')\nassert not is_palindrome('abc')\n"},
    {"id": "fib",
     "prompt": "Write a complete python function fib(n) returning the n-th Fibonacci number "
               "(fib(0)=0, fib(1)=1). Only code.",
     "check": "assert fib(0) == 0 and fib(1) == 1 and fib(7) == 13\n"},
]

_BUILTIN_PPL = [
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "Compilers translate source code into machine instructions through several passes.",
    "The treaty was signed in the spring, ending a conflict that had lasted three years.",
]


def load_code_problems(n: int = 20, source: str = "builtin") -> list[dict]:
    """Code-execution problems as ``{id, prompt, prefix, check}``.

    For HumanEval the model completes a given signature, so the signature+docstring must be in the
    executed program: ``prefix`` (the prompt) is prepended to the model output in the sandbox, and
    ``check`` runs the dataset's ``check(candidate)`` against the ``entry_point``.
    """
    if source != "real":
        return [{**p, "prefix": ""} for p in _BUILTIN_CODE[:n]]
    from datasets import load_dataset
    d = load_dataset("openai/openai_humaneval", split=f"test[:{n}]")
    return [{
        "id": r["task_id"],
        "prompt": r["prompt"],
        "prefix": r["prompt"],                       # signature+docstring -> into the program
        "check": r["test"] + f"\ncheck({r['entry_point']})\n",
    } for r in d]


def load_ppl_texts(n: int = 20, source: str = "builtin") -> list[str]:
    """Held-out text for perplexity."""
    if source != "real":
        return _BUILTIN_PPL[:n]
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t.strip() for t in d["text"] if len(t.strip()) > 200]
    return texts[:n]


def load_tool_problems(n: int = 20, source: str = "builtin") -> list[dict]:
    """Single-call function-calling problems as ``{id, prompt, gold}`` (bundled; see module doc)."""
    with open(os.path.join(_DATA_DIR, "tool_calls.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n]
