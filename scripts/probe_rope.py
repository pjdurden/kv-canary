"""Local de-risk probe: does cache-level eviction produce COHERENT output on a RoPE model?

distilgpt2 (learned absolute positions) can't expose the post-eviction re-positioning issue.
Qwen2.5 uses RoPE, so this is the cheap CPU canary for whether the GPU run will be valid or
confounded by broken positions. Run before spending GPU money.
"""
import sys
from kvcanary.backends.hf import HFBackend
from kvcanary.compressors.full import FullCache
from kvcanary.compressors.streamingllm import StreamingLLMCompressor
from kvcanary.compressors.snapkv import SnapKVCompressor

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = ("You are a helpful assistant. The capital of France is Paris, the capital of Japan is "
          "Tokyo, and the capital of Italy is Rome. Question: what is the capital of Japan? Answer:")

b = HFBackend(model_id=MODEL, device="cpu")
print("loaded", MODEL, "| rope_theta:", getattr(b.model.config, "rope_theta", "n/a"))

for name, comp in [("full", FullCache(1.0)),
                   ("streamingllm@0.25", StreamingLLMCompressor(0.25)),
                   ("snapkv@0.25", SnapKVCompressor(0.25))]:
    g = b.generate(PROMPT, compressor=comp, max_new_tokens=24)
    ppl = b.score(PROMPT, compressor=comp)
    print(f"\n[{name}] ppl={ppl:.2f}")
    print("  out:", repr(g.text[:160]))
    sys.stdout.flush()
