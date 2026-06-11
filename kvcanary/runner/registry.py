from kvcanary.compressors.full import FullCache
from kvcanary.compressors.streamingllm import StreamingLLMCompressor
from kvcanary.compressors.snapkv import SnapKVCompressor
from kvcanary.compressors.quantized import QuantizedCache


def build_compressor(method: str, budget: float):
    if method == "full":
        return FullCache(budget)
    if method == "streamingllm":
        return StreamingLLMCompressor(budget)
    if method == "snapkv":
        return SnapKVCompressor(budget)
    if method == "quantized":
        return QuantizedCache(bits=round(budget * 16))
    raise ValueError(f"unknown method {method}")
