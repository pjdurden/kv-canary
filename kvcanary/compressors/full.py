from kvcanary.compressors.base import KVCompressor


class FullCache(KVCompressor):
    name = "full"
    family = "none"
