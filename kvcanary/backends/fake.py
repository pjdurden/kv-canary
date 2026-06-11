from kvcanary.backends.base import Backend, Generation


class FakeBackend(Backend):
    def __init__(self, scripted: dict[str, str], ppl: float = 10.0, full_bytes: float = 1000.0):
        self.scripted = scripted
        self.ppl = ppl
        self.full_bytes = full_bytes

    def generate(self, prompt, compressor, max_new_tokens=256) -> Generation:
        text = self.scripted.get(prompt, "")
        return Generation(
            text=text,
            n_tokens=len(text.split()),
            latency_s=0.0,
            kv_bytes_retained=compressor.kv_bytes_retained(self.full_bytes),
        )

    def score(self, text, compressor) -> float:
        return self.ppl
