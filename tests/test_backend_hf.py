import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from kvcanary.backends.hf import HFBackend
from kvcanary.compressors.full import FullCache

@pytest.mark.slow
def test_hf_backend_deterministic_and_perplexity_sane():
    b = HFBackend(model_id="distilgpt2", device="cpu")
    c = FullCache(1.0)
    o1 = b.generate("The capital of France is", compressor=c, max_new_tokens=5)
    o2 = b.generate("The capital of France is", compressor=c, max_new_tokens=5)
    assert o1.text == o2.text                       # greedy determinism
    p_clean = b.score("the cat sat on the mat", compressor=c)
    p_scram = b.score("mat the on sat cat the", compressor=c)
    assert p_clean > 0 and p_scram > p_clean        # scrambled text is less likely

@pytest.mark.slow
def test_hf_backend_exposes_full_bytes():
    b = HFBackend(model_id="distilgpt2", device="cpu")
    assert b.full_bytes > 0                          # representative full-KV size for ppl accounting
