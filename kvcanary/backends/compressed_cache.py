"""A ``DynamicCache`` that actually applies KV compression during generation.

This is the piece that turns the harness from "records how much KV each method *would*
retain" into "produces the divergence" — i.e. lossy methods now change the model's output.

It dispatches on ``compressor.family``:

* ``quant`` (QuantizedCache) — every appended key/value tensor is passed through
  ``compressor.quantize`` before it enters the cache, so the *whole* cache is stored lossy.
  Applied every step; exact.
* ``evict`` + score-free (StreamingLLM) — the prompt KV is evicted to a fixed kept-token set
  inside ``update`` at prefill, using ``compressor.kept_indices`` (positional sink + recent).
* ``evict`` + attention-based (SnapKV, ``needs_attention``) — eviction needs the prefill
  attention scores, which ``update`` does not see. The prompt is left intact until the backend
  calls :meth:`compress_prompt` with the attentions captured from a one-time prefill forward.

Design choices (kept honest on purpose — this is a *correctness* harness):

* **Budget is resolved once, against the prompt length** (``keep = round(budget * n_prompt)``),
  giving a fixed kept-token count. These methods are prompt-compression methods, so decode
  tokens append on top of the kept prompt and are not themselves re-evicted in v1. (StreamingLLM
  technically keeps a *rolling* recent window during decode; omitting that roll is a v1
  simplification — the divergence-driving loss is the prompt compression, and these tasks emit
  short outputs. Documented, not silent.)
* **The current step still attends over the full tensors.** ``update`` returns the un-evicted
  keys/values for *this* forward; only the stored cache is shrunk, so the prefill attention is
  exact and only subsequent (decode) steps see the kept subset.

VALIDATE ON THE GPU RUN (two known fidelity points, both fine on GPT2 / distilgpt2 which uses
*learned absolute* positions, but must be checked on RoPE models like Qwen):
  1. **Re-positioning.** HF caches keys *post-RoPE*. After eviction the kept keys carry their
     original rotary phase, while the next decode token is placed at position
     ``cache.get_seq_length() == keep`` — a gap vs its true prompt position. True StreamingLLM
     re-ropes the kept keys to contiguous positions; that is not done here. Confirm the effect
     (and whether it helps or hurts) on Qwen before trusting absolute numbers.
  2. **cache_position / mask.** Confirm the model derives the decode mask from the *post-eviction*
     cache length (it should, via ``get_mask_sizes``), so no attention to dropped slots leaks in.
"""
import torch
from transformers import DynamicCache


class CompressedCache(DynamicCache):
    def __init__(self, compressor, *, config=None):
        super().__init__(config=config)
        self.compressor = compressor
        self.prompt_len = None        # resolved at prefill (first multi-token update)
        self.keep = None              # fixed kept-token count = round(budget * prompt_len)
        self._pending_attn = bool(getattr(compressor, "needs_attention", False))
        self._evicted = set()         # layer indices already compressed (evict the context once)

    @property
    def compressed(self) -> bool:
        """True once the context has been evicted (any layer). Evict-once: a later multi-token
        forward — e.g. the teacher-forced continuation in perplexity scoring — must NOT re-evict."""
        return len(self._evicted) > 0

    # ------------------------------------------------------------------ update
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        fam = getattr(self.compressor, "family", "none")

        if fam == "quant":
            # Store the whole cache lossily: quantize the states being appended this step.
            key_states = self.compressor.quantize(key_states)
            value_states = self.compressor.quantize(value_states)
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if fam != "evict":
            return keys, values

        is_prefill = key_states.shape[-2] > 1
        if is_prefill and self.keep is None:
            self.prompt_len = keys.shape[-2]
            self.keep = round(self.compressor.budget * self.prompt_len)

        # Score-free eviction (StreamingLLM): apply once per layer, at the context prefill. SnapKV
        # waits for compress_prompt() (needs attention). The `not in self._evicted` guard makes a
        # later multi-token forward (the perplexity continuation block) append without re-evicting.
        if is_prefill and not self._pending_attn and layer_idx not in self._evicted:
            self._evict_layer(layer_idx, self.prompt_len, attn_scores=None)
            self._evicted.add(layer_idx)

        # Return the full (pre-eviction) tensors so THIS step's attention is exact; the stored
        # layer cache is already the kept subset for the next step.
        return keys, values

    # --------------------------------------------------------- attention path
    @torch.no_grad()
    def compress_prompt(self, attentions):
        """Evict the prompt using prefill attention (SnapKV). ``attentions`` is the
        ``output_attentions=True`` tuple: one ``[batch, heads, q_len, kv_len]`` tensor per layer.
        No-op for methods that don't need attention or once already compressed.
        """
        if not self._pending_attn or self.compressed:
            return
        window = getattr(self.compressor, "window", 8)
        for layer_idx, attn in enumerate(attentions):
            # Observation window = last `window` query rows attending over the prefix; mean over
            # heads and over the window gives a per-position importance score (length = prompt_len).
            scores = attn[0, :, -window:, :].mean(dim=(0, 1)).to(torch.float32)
            self._evict_layer(layer_idx, self.prompt_len, attn_scores=scores)
            self._evicted.add(layer_idx)
        self._pending_attn = False

    # ---------------------------------------------------------------- helpers
    def _evict_layer(self, layer_idx, n_tokens, attn_scores):
        idx = self.compressor.kept_indices(n_tokens, attn_scores=attn_scores)
        if len(idx) >= n_tokens:
            return  # nothing to drop
        layer = self.layers[layer_idx]
        sel = torch.as_tensor(sorted(idx), device=layer.keys.device, dtype=torch.long)
        layer.keys = layer.keys.index_select(-2, sel).contiguous()
        layer.values = layer.values.index_select(-2, sel).contiguous()
