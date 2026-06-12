import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvcanary.backends.base import Backend, Generation
from kvcanary.backends.compressed_cache import CompressedCache


class HFBackend(Backend):
    """Real HuggingFace causal-LM backend.

    Compression is now wired into HF's KV cache via :class:`CompressedCache`. The ``full``
    baseline uses HF's fast ``model.generate`` path unchanged; lossy methods (quant / eviction)
    run a controlled prefill+greedy-decode loop so the compressed cache actually shapes the
    output. See ``compressed_cache.py`` for the per-family semantics and the two RoPE fidelity
    points to validate on the GPU run.
    """

    def __init__(self, model_id: str, device: str = "cpu", dtype=None):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # eager attention so SnapKV's prefill `output_attentions=True` actually returns scores
        # (sdpa/flash don't). Default fp16 on CUDA so a 7B fits a 24GB card; fp32 on CPU.
        if dtype is None:
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
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
        n_prompt = ids.input_ids.shape[1]

        if getattr(compressor, "family", "none") == "none":
            # Baseline: HF's fast path, identical to a normal run (no cache surgery).
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            new_ids = out[0, n_prompt:]
        else:
            new_ids = self._compressed_generate(ids, compressor, max_new_tokens)

        text = self.tok.decode(new_ids, skip_special_tokens=True)
        seq_len = n_prompt + new_ids.shape[0]
        return Generation(text=text, n_tokens=int(new_ids.shape[0]),
                          latency_s=time.perf_counter() - t0,
                          kv_bytes_retained=compressor.kv_bytes_retained(self._full_kv_bytes(seq_len)))

    @torch.no_grad()
    def _compressed_generate(self, ids, compressor, max_new_tokens):
        """Greedy decode through a CompressedCache so lossy KV actually shapes the output."""
        cache = CompressedCache(compressor, config=self.model.config)
        needs_attn = bool(getattr(compressor, "needs_attention", False))

        # Prefill. Capture attentions only when the method needs them (forces eager attention
        # for this one forward — fine for a single prefill; set attn_implementation="eager" on
        # large RoPE models if they default to sdpa/flash).
        out = self.model(input_ids=ids.input_ids, attention_mask=ids.attention_mask,
                         past_key_values=cache, use_cache=True, output_attentions=needs_attn)
        if needs_attn:
            cache.compress_prompt(out.attentions)

        next_id = out.logits[:, -1:, :].argmax(-1)        # [1, 1]
        collected = [next_id]
        eos = self.tok.eos_token_id
        for _ in range(max_new_tokens - 1):
            if eos is not None and next_id.item() == eos:
                break
            out = self.model(input_ids=next_id, past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1:, :].argmax(-1)
            collected.append(next_id)
        return torch.cat(collected, dim=-1)[0]

    @torch.no_grad()
    def score(self, text, compressor, ctx_frac: float = 0.5) -> float:
        """Perplexity of a continuation under a (possibly compressed) context KV cache.

        The text is split into a context (first ``ctx_frac``) and a continuation (the rest). The
        context is prefilled through a :class:`CompressedCache`, so the compressor actually shapes
        what the continuation attends to; the continuation is then teacher-forced and its mean NLL
        becomes the perplexity. This is what lets the perplexity LINE move under compression — the
        old path scored the uncompressed model and was flat by construction, making the divergence
        circular. EVERY method (including ``full``) uses this same context/continuation protocol,
        so perplexities are measured over identical tokens and differ only by the context
        compression — that difference is the whole point. Very short texts (no room to split) fall
        back to plain whole-text perplexity, identically for all methods.
        """
        ids = self.tok(text, return_tensors="pt").to(self.device)
        input_ids = ids.input_ids
        n_tok = input_ids.shape[1]
        if n_tok < 8:
            out = self.model(input_ids=input_ids, labels=input_ids)
            return float(torch.exp(out.loss))

        c = max(1, int(n_tok * ctx_frac))
        ctx_ids, cont_ids = input_ids[:, :c], input_ids[:, c:]
        cache = CompressedCache(compressor, config=self.model.config)
        needs_attn = bool(getattr(compressor, "needs_attention", False))

        out_ctx = self.model(input_ids=ctx_ids, attention_mask=ids.attention_mask[:, :c],
                             past_key_values=cache, use_cache=True, output_attentions=needs_attn)
        if needs_attn:
            cache.compress_prompt(out_ctx.attentions)

        out_cont = self.model(input_ids=cont_ids, past_key_values=cache, use_cache=True)
        # Logits predicting each continuation token: the last context logit predicts cont[0], then
        # cont logits 0..L-2 predict cont 1..L-1.
        logits = torch.cat([out_ctx.logits[:, -1:, :], out_cont.logits[:, :-1, :]], dim=1)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), cont_ids.reshape(-1), reduction="mean")
        return float(torch.exp(nll))
