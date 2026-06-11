import torch
from kvcanary.compressors.base import KVCompressor


class SnapKVCompressor(KVCompressor):
    name = "snapkv"
    family = "evict"

    def __init__(self, budget: float, window: int = 8):
        super().__init__(budget)
        self.window = window

    def kept_indices(self, n_tokens: int, attn_scores=None) -> list[int]:
        keep = round(self.budget * n_tokens)
        if keep >= n_tokens:
            return list(range(n_tokens))
        if attn_scores is None:
            raise ValueError("SnapKV needs attn_scores to select tokens")
        win_start = max(0, n_tokens - self.window)
        window_idx = set(range(win_start, n_tokens))
        budget_left = max(0, keep - len(window_idx))
        cand = attn_scores.clone()
        cand[win_start:] = float("-inf")          # exclude window from ranking
        top = torch.topk(cand, k=min(budget_left, win_start)).indices.tolist()
        return sorted(window_idx | set(top))

    def kv_bytes_retained(self, full_bytes: float) -> float:
        return full_bytes * self.budget
