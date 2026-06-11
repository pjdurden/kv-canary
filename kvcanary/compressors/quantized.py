import torch
from kvcanary.compressors.base import KVCompressor


class QuantizedCache(KVCompressor):
    name = "quantized"
    family = "quant"

    def __init__(self, bits: int = 4):
        super().__init__(budget=bits / 16.0)
        self.bits = bits

    def quantize(self, tensor):
        qmax = 2 ** (self.bits - 1) - 1
        scale = tensor.abs().max().clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax)
        return q * scale

    def kv_bytes_retained(self, full_bytes: float) -> float:
        return full_bytes * self.budget
