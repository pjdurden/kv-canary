from abc import ABC, abstractmethod
from kvcanary.types import Sample, EvalResult

class Task(ABC):
    name: str = "base"
    needs_generation: bool = True   # False for perplexity-only
    @abstractmethod
    def build_samples(self) -> list[Sample]: ...
    @abstractmethod
    def evaluate(self, sample: Sample, output: str, **kw) -> EvalResult: ...
