import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kvcanary.backends.compressed_cache import CompressedCache
from kvcanary.compressors.streamingllm import StreamingLLMCompressor
from kvcanary.compressors.snapkv import SnapKVCompressor
from kvcanary.compressors.quantized import QuantizedCache


def _kv(n_tokens, heads=4, head_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (1, heads, n_tokens, head_dim)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


# ----- model-free unit tests: the eviction/quant logic itself -----

def test_streamingllm_evicts_prompt_at_prefill():
    cache = CompressedCache(StreamingLLMCompressor(0.5))
    k, v = _kv(10)
    keys, _ = cache.update(k, v, layer_idx=0)
    # current step sees the FULL prompt (exact attention)...
    assert keys.shape[-2] == 10
    # ...but the stored cache is shrunk to round(0.5 * 10) = 5 for the next step.
    assert cache.layers[0].keys.shape[-2] == 5
    assert cache.keep == 5 and cache.compressed


def test_decode_token_appends_on_kept_prompt():
    cache = CompressedCache(StreamingLLMCompressor(0.5))
    cache.update(*_kv(10), layer_idx=0)            # prefill -> kept 5
    k1, v1 = _kv(1, seed=1)
    cache.update(k1, v1, layer_idx=0)              # one decode step
    assert cache.layers[0].keys.shape[-2] == 6     # 5 kept prompt + 1 generated


def test_quantized_stores_lossy_but_same_shape():
    cache = CompressedCache(QuantizedCache(bits=4))
    k, v = _kv(6)
    cache.update(k, v, layer_idx=0)
    stored = cache.layers[0].keys
    assert stored.shape == k.shape                 # no eviction, shape preserved
    assert not torch.equal(stored, k)              # but values were quantized (lossy)
    # int4 round-trip error is bounded by the quant step (max/qmax)
    assert (stored - k).abs().max() <= k.abs().max() / 7 + 1e-6


def test_snapkv_waits_for_attention_then_evicts():
    # prompt length 40 (> SnapKV window 8) so keep=20 exceeds the window and the budget holds.
    cache = CompressedCache(SnapKVCompressor(0.5))
    cache.update(*_kv(40), layer_idx=0)
    # SnapKV needs attention -> prompt NOT compressed inside update()
    assert cache.layers[0].keys.shape[-2] == 40
    assert not cache.compressed
    # last-window attention over the 40 prefix positions, one layer
    attn = torch.rand(1, 4, 40, 40)
    cache.compress_prompt((attn,))
    assert cache.layers[0].keys.shape[-2] == 20
    assert cache.compressed


# ----- end-to-end: lossy KV actually changes the output (divergence != NULL) -----

def test_continuation_block_does_not_re_evict():
    # Perplexity scoring runs a SECOND multi-token forward (the continuation). Evict-once must
    # leave the kept context intact and simply append the continuation tokens.
    cache = CompressedCache(StreamingLLMCompressor(0.5))
    cache.update(*_kv(10), layer_idx=0)            # context prefill -> kept 5
    assert cache.layers[0].keys.shape[-2] == 5
    cache.update(*_kv(4, seed=2), layer_idx=0)     # continuation block of 4 tokens
    assert cache.layers[0].keys.shape[-2] == 9     # 5 kept context + 4 continuation, no re-evict


# Ends with "In summary," — a cue whose continuation depends on the EARLY context that aggressive
# eviction drops, so the lossy method visibly diverges (greedy next-tokens aren't purely local).
_PROSE = ("Linkers and loaders translate symbolic references into addresses. The relocation "
          "table records each fixup the loader must apply at load time, so that position "
          "independent code can run from any base address. In summary,")


@pytest.mark.slow
def test_eviction_changes_output_vs_full():
    from kvcanary.backends.hf import HFBackend
    from kvcanary.compressors.full import FullCache

    b = HFBackend(model_id="distilgpt2", device="cpu")
    full = b.generate(_PROSE, compressor=FullCache(1.0), max_new_tokens=20)
    evicted = b.generate(_PROSE, compressor=StreamingLLMCompressor(0.25), max_new_tokens=20)

    # The whole point: aggressive prompt eviction perturbs the greedy continuation.
    # (If this ever fails, the cache wiring regressed back to a no-op and divergence is NULL.)
    assert full.text != evicted.text
    assert evicted.n_tokens > 0


@pytest.mark.slow
def test_score_responds_to_compression():
    from kvcanary.backends.hf import HFBackend
    from kvcanary.compressors.full import FullCache

    b = HFBackend(model_id="distilgpt2", device="cpu")
    p_full = b.score(_PROSE, compressor=FullCache(1.0))
    p_evict = b.score(_PROSE, compressor=StreamingLLMCompressor(0.25))

    # score() now measures continuation perplexity under the (compressed) context, so a lossy
    # method must NOT equal the full baseline. The old path ignored the compressor -> identical
    # numbers -> a perplexity line flat by construction (the circular finding this fixes).
    assert p_full > 0 and p_evict > 0
    assert p_evict != p_full
