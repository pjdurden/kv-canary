import torch
from kvcanary.compressors.full import FullCache
from kvcanary.compressors.streamingllm import StreamingLLMCompressor
from kvcanary.compressors.snapkv import SnapKVCompressor


def test_full_cache_is_noop_and_retains_all_memory():
    c = FullCache(budget=1.0)
    n = 10
    idx = c.kept_indices(n_tokens=n, attn_scores=None)
    assert idx == list(range(n))               # keeps every token
    assert c.kv_bytes_retained(full_bytes=1000) == 1000  # fp16 baseline, no saving
    assert c.name == "full"


def test_streamingllm_keeps_sinks_plus_recent():
    c = StreamingLLMCompressor(budget=0.5, n_sink=2)
    idx = c.kept_indices(n_tokens=10, attn_scores=None)   # keep ~5: 2 sinks + 3 recent
    assert idx[:2] == [0, 1]                # sinks
    assert idx[-1] == 9 and len(idx) == 5   # most recent + correct count
    assert sorted(idx) == idx               # sorted, no dups


def test_streamingllm_budget_one_equals_full():
    c = StreamingLLMCompressor(budget=1.0, n_sink=2)
    assert c.kept_indices(n_tokens=10) == list(range(10))   # invariant: keep-all == full


def test_streamingllm_bytes_scale_with_budget():
    c = StreamingLLMCompressor(budget=0.25, n_sink=2)
    assert c.kv_bytes_retained(full_bytes=1000) == 250


def test_streamingllm_recent_zero_keeps_sinks_only():
    # When keep <= n_sink (e.g. low budget), recent window is empty -> sinks only.
    c = StreamingLLMCompressor(budget=0.2, n_sink=2)
    idx = c.kept_indices(n_tokens=10)   # keep = round(0.2*10) = 2 == n_sink -> recent 0
    assert idx == [0, 1]


def test_snapkv_selects_high_attention_tokens():
    c = SnapKVCompressor(budget=0.5, window=2)
    # 8 tokens; last 2 are the observation window (always kept).
    # Among the first 6, tokens 1 and 4 have the highest attention.
    scores = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.0, 0.0])
    idx = c.kept_indices(n_tokens=8, attn_scores=scores)  # keep 4: window {6,7} + top-2 {1,4}
    assert set(idx) == {1, 4, 6, 7}
    assert sorted(idx) == idx


def test_snapkv_respects_budget_when_keep_below_window():
    # Regression: at low budget keep can be < window. The window must not blow the budget.
    c = SnapKVCompressor(budget=0.25, window=8)
    scores = torch.arange(20, dtype=torch.float)          # token 19 highest, descends
    idx = c.kept_indices(n_tokens=20, attn_scores=scores)  # keep = round(0.25*20) = 5
    assert len(idx) == 5                                   # exactly the budget, not `window`
    assert sorted(idx) == idx
    assert max(idx) == 19                                  # most-recent observation token kept


def test_snapkv_budget_one_equals_full():
    c = SnapKVCompressor(budget=1.0, window=2)
    assert c.kept_indices(n_tokens=8, attn_scores=torch.zeros(8)) == list(range(8))


def test_snapkv_requires_scores():
    c = SnapKVCompressor(budget=0.5, window=2)
    try:
        c.kept_indices(n_tokens=8, attn_scores=None)
        assert False, "expected ValueError"
    except ValueError:
        pass


from kvcanary.compressors.quantized import QuantizedCache

def test_quant_is_lossy_but_bounded_and_keeps_all_tokens():
    c = QuantizedCache(bits=4)
    x = torch.randn(64)
    y = c.quantize(x)
    assert y.shape == x.shape
    assert torch.any(y != x)                     # genuinely lossy
    assert (x - y).abs().max() < x.abs().max()   # bounded by step size
    assert c.kept_indices(n_tokens=10) == list(range(10))   # quant evicts nothing

def test_quant_budget_and_bytes():
    c = QuantizedCache(bits=4)
    assert c.budget == 0.25
    assert c.kv_bytes_retained(full_bytes=1600) == 400      # 4/16 of fp16
