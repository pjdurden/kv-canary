import torch
from kvcanary.compressors.base import KVCompressor


class SnapKVCompressor(KVCompressor):
    name = "snapkv"
    family = "evict"
    needs_attention = True

    def __init__(self, budget: float, window: int = 8):
        super().__init__(budget)
        self.window = window

    def kept_indices(self, n_tokens: int, attn_scores=None) -> list[int]:
        keep = round(self.budget * n_tokens)
        if keep >= n_tokens:
            return list(range(n_tokens))
        if attn_scores is None:
            raise ValueError("SnapKV needs attn_scores to select tokens")
        # The observation window can't exceed the budget, or we'd keep more than `keep`
        # tokens at low budgets (keep < window) and silently overshoot the memory target.
        eff_window = min(self.window, keep)
        win_start = n_tokens - eff_window
        window_idx = set(range(win_start, n_tokens))
        budget_left = keep - len(window_idx)      # >= 0 now that the window is capped
        cand = attn_scores.clone()
        cand[win_start:] = float("-inf")          # exclude window from ranking
        top = torch.topk(cand, k=min(budget_left, win_start)).indices.tolist()
        return sorted(window_idx | set(top))

    def kv_bytes_retained(self, full_bytes: float) -> float:
        return full_bytes * self.budget
