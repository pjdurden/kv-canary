from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    n_tokens: int
    latency_s: float
    kv_bytes_retained: float


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, compressor, max_new_tokens: int = 256) -> Generation:
        ...

    @abstractmethod
    def score(self, text: str, compressor) -> float:
        ...
