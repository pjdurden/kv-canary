from kvcanary.backends.fake import FakeBackend
from kvcanary.compressors.full import FullCache


def test_fake_backend_generate_and_score():
    b = FakeBackend(scripted={"hello": "world"}, ppl=12.5)
    c = FullCache(budget=1.0)
    out = b.generate("hello", compressor=c, max_new_tokens=8)
    assert out.text == "world" and out.n_tokens == 1 and out.kv_bytes_retained > 0
    assert b.score("anything", compressor=c) == 12.5
