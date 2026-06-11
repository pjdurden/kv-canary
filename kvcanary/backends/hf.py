import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcanary.backends.base import Backend, Generation


class HFBackend(Backend):
    """Real HuggingFace causal-LM backend.

    V1 DEFERRAL: compressors are NOT wired into HF's KV cache here. Generation runs normal
    greedy decoding; only ``kv_bytes_retained`` (reported via the compressor) differs across
    methods. All methods produce IDENTICAL text and perplexity until the DynamicCache-subclass
    eviction wiring lands (Task 16, on the GPU run). Do NOT interpret zero divergence in the v1
    matrix with this backend as evidence that the compression methods have no effect.
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
        self.device = device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Representative full-KV byte size for perplexity-row accounting (the runner reads
        # backend.full_bytes for the score path). Nominal 512-token cache; the downstream
        # x-axis uses the retained FRACTION (budget), so the absolute baseline only needs to
        # be consistent, not exact.
        self.full_bytes = self._full_kv_bytes(512)

    def _full_kv_bytes(self, seq_len: int) -> float:
        cfg = self.model.config
        layers = cfg.num_hidden_layers
        heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        return 2 * layers * heads * head_dim * seq_len * 2  # 2 tensors (K,V) * fp16 (2 bytes)

    @torch.no_grad()
    def generate(self, prompt, compressor, max_new_tokens=256) -> Generation:
        t0 = time.perf_counter()
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        seq_len = out.shape[1]
        full_bytes = self._full_kv_bytes(seq_len)
        return Generation(text=text, n_tokens=out.shape[1] - ids.input_ids.shape[1],
                          latency_s=time.perf_counter() - t0,
                          kv_bytes_retained=compressor.kv_bytes_retained(full_bytes))

    @torch.no_grad()
    def score(self, text, compressor) -> float:
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model(**ids, labels=ids.input_ids)
        return float(torch.exp(out.loss))
