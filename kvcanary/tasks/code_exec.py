import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from kvcanary.tasks.base import Task
from kvcanary.types import Sample, EvalResult

@dataclass
class SandboxResult:
    ok: bool
    stderr: str

def run_in_sandbox(code: str, timeout: int = 10) -> SandboxResult:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snip.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout, cwd=d,
                               # bare env drops PATH/PYTHONPATH; cwd=d keeps any file writes
                               # inside the temp dir. Not adversarial isolation — eval-grade only.
                               env={"PATH": "", "PYTHONPATH": ""})
            return SandboxResult(ok=(p.returncode == 0), stderr=p.stderr)
        except subprocess.TimeoutExpired:
            # subprocess.run kills the direct child only; grandchildren spawned by the
            # snippet can outlive the timeout. Acceptable for non-adversarial eval code.
            return SandboxResult(ok=False, stderr="timeout")

class CodeExecTask(Task):
    name = "code"
    def __init__(self, problems: list[dict]):
        self.problems = problems
    def build_samples(self):
        # prefix (e.g. the HumanEval signature+docstring) is prepended to the model output before
        # the check, so completion-style problems form a runnable program. Defaults to "".
        return [Sample(id=p["id"], prompt=p["prompt"],
                       meta={"check": p["check"], "prefix": p.get("prefix", "")})
                for p in self.problems]
    def evaluate(self, sample, output: str, **kw):
        program = sample.meta.get("prefix", "") + output + "\n" + sample.meta["check"]
        res = run_in_sandbox(program, timeout=10)
        return EvalResult(scores={"pass@1": 1.0 if res.ok else 0.0},
                          detail={"stderr": res.stderr[:500]})
