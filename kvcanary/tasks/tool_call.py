import json
import re
from kvcanary.tasks.base import Task
from kvcanary.types import Sample, EvalResult


class ToolCallTask(Task):
    name = "tool"

    def __init__(self, problems: list[dict]):
        self.problems = problems

    def build_samples(self):
        return [
            Sample(id=p["id"], prompt=p["prompt"], meta={"gold": p["gold"]})
            for p in self.problems
        ]

    def evaluate(self, sample, output: str, **kw):
        gold = sample.meta["gold"]
        scores = {"json_valid": 0.0, "func_correct": 0.0, "arg_match": 0.0}
        m = re.search(r"\{.*\}", output, re.DOTALL)  # greedy: first { to last } (single-call outputs)
        if not m:
            return EvalResult(scores=scores)
        try:
            call = json.loads(m.group(0))
        except json.JSONDecodeError:
            return EvalResult(scores=scores)
        scores["json_valid"] = 1.0
        scores["func_correct"] = 1.0 if call.get("name") == gold["name"] else 0.0
        scores["arg_match"] = (
            1.0 if call.get("arguments") == gold["arguments"] else 0.0
        )
        return EvalResult(scores=scores)
