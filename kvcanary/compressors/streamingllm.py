from kvcanary.compressors.base import KVCompressor


class StreamingLLMCompressor(KVCompressor):
    name = "streamingllm"
    family = "evict"

    def __init__(self, budget: float, n_sink: int = 4):
        super().__init__(budget)
        self.n_sink = n_sink

    def kept_indices(self, n_tokens: int, attn_scores=None) -> list[int]:
        keep = round(self.budget * n_tokens)
        if keep >= n_tokens:
            return list(range(n_tokens))
        sink = min(self.n_sink, n_tokens, keep)
        recent = keep - sink
        idx = list(range(sink)) + list(range(n_tokens - recent, n_tokens))
        return sorted(set(i for i in idx if 0 <= i < n_tokens))

    def kv_bytes_retained(self, full_bytes: float) -> float:
        return full_bytes * self.budget
