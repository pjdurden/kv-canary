from abc import ABC


class KVCompressor(ABC):
    name: str = "base"
    family: str = "none"      # "none" | "evict" | "quant"
    needs_attention: bool = False   # True if kept_indices requires attn_scores (SnapKV)

    def __init__(self, budget: float):
        self.budget = float(budget)

    def kept_indices(self, n_tokens: int, attn_scores=None) -> list[int]:
        """Eviction: indices of tokens to KEEP. Default keeps all."""
        return list(range(n_tokens))

    def quantize(self, tensor):
        """Quantization: return (possibly lossy) tensor. Default identity."""
        return tensor

    def kv_bytes_retained(self, full_bytes: float) -> float:
        """Bytes of KV retained vs the fp16 full cache (the shared x-axis numerator)."""
        return full_bytes
