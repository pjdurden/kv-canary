from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    scores: dict[str, float]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CellSpec:
    model: str
    method: str
    budget: float
    task: str

    # NOTE: cell_id is the single source of truth for a cell's string id. budget is always a
    # float; any code needing this id must reuse this property (not rebuild the string), so the
    # JSONL written by CellResult.to_row() and the runner's resume check stay byte-identical.
    @property
    def cell_id(self) -> str:
        return f"{self.model}|{self.method}|{self.budget}|{self.task}"


@dataclass
class CellResult:
    spec: CellSpec
    sample_id: str
    scores: dict[str, float]
    raw_output: str
    n_tokens: int
    latency_s: float
    kv_bytes_retained: float

    def to_row(self) -> dict:
        return {
            "model": self.spec.model,
            "method": self.spec.method,
            "budget": self.spec.budget,
            "task": self.spec.task,
            "cell_id": self.spec.cell_id,
            "sample_id": self.sample_id,
            "scores": self.scores,
            "raw_output": self.raw_output,
            "n_tokens": self.n_tokens,
            "latency_s": self.latency_s,
            "kv_bytes_retained": self.kv_bytes_retained,
        }
