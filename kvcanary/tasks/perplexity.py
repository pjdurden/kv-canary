from kvcanary.tasks.base import Task
from kvcanary.types import Sample, EvalResult

class PerplexityTask(Task):
    name = "perplexity"
    needs_generation = False
    def __init__(self, texts: list[str]):
        self.texts = texts
    def build_samples(self):
        return [Sample(id=f"ppl{i}", prompt=t) for i, t in enumerate(self.texts)]
    def evaluate(self, sample, output="", ppl: float = float("nan"), **kw):
        return EvalResult(scores={"perplexity": ppl})
