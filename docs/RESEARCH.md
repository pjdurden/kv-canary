# kv-canary — foundations & gap audit (deep research, 2026-06-12)

A multi-source, adversarially-verified review of the claims and method implementations behind
kv-canary. 23 sources fetched, 99 claims extracted, 25 verified by 3-vote adversarial checking
(24 confirmed, 1 killed). Load-bearing arXiv IDs were independently re-fetched. Findings that
change the project are recorded here so the repo is honest about what it does and doesn't establish.

## TL;DR verdict

- The **thesis is sound and correctly grounded**: lossy KV-cache compression really does cause
  catastrophic functional failures in code generation and tool calling (VeriCache, real).
- The **originality wedge is false**: functional/downstream degradation under KV compression is
  already measured by NVIDIA KVPress and multiple 2025–2026 papers. Only a *narrow* corner survives.
- **Two method implementations are wrong**, not merely simplified: StreamingLLM without re-roping,
  and head-averaged SnapKV without pooling. A knowledgeable reviewer would dismiss those results.
- The **prompt-only design may be confounded** for the long-decode failure mode it targets.

## Q1 — Citation check

**VeriCache is real.** `arXiv:2605.17613` = *"VeriCache: Turning Lossy KV Cache into Lossless LLM
Inference"* (Yao, Shen, Du, Feng, Seo, Zhang, Huang, Liu, Lu, Jiang; May 17 2026). Abstract:
> "almost all of these methods are inherently lossy—despite minimal accuracy degradation for short
> outputs, their outputs increasingly diverge from full-KV-cache outputs as more tokens are decoded,
> which leads to catastrophic failures in code generation and tool calling."

**But VeriCache does NOT make the "perplexity is blind" argument** (verified, 0-3 — the one killed
claim). It never mentions perplexity; it frames short-output-small vs long-output-large divergence.
Citing VeriCache for "token-level metrics stay flat" is a **misattribution**.

The closest real support for "aggregate/token-level metrics hide functional damage" is
***The Pitfalls of KV Cache Compression*** (`arXiv:2510.00231`, Chen, Geh, Grover, Van den Broeck,
Israel; UCLA; ACL 2026):
> "certain instructions degrade much more rapidly with compression, effectively causing them to be
> completely ignored by the LLM."

It evaluates the exact methods kv-canary models (StreamingLLM, SnapKV, +TOVA/H2O/K-Norm) on
Llama3.1-8B and Qwen2.5-14B — but measures **IFEval instruction-following, not perplexity**. So
"perplexity fails to warn" is kv-canary's **hypothesis to test**, not an established result. Source:
https://arxiv.org/abs/2605.17613 · https://arxiv.org/abs/2510.00231

## Q2 — Is the wedge novel? No (broadly)

"Everyone measures speed/memory, nobody measures functional correctness under KV compression" is
**contradicted**:
- **NVIDIA KVPress** (https://github.com/NVIDIA/kvpress, maintained) — accuracy CLI over RULER,
  LongBench/-v2, Loogle, InfiniteBench, Zero-Scrolls, Needle-in-a-Haystack.
- **Pitfalls** (2510.00231) — IFEval instruction-following.
- **Hold Onto That Thought** (`arXiv:2512.12008`, Dec 2025) — 8 reasoning benchmarks (FOLIO, DROP,
  GSM8K, MATH-500, ReClor, StrategyQA, CommonSenseQA, OpenBookQA), built on the kvpress library.

**Narrow survivor:** none of these score **executable-code-passing-unit-tests + valid-JSON/tool-call
schema conformance** specifically (they do QA / retrieval / reasoning / instruction-following
accuracy). That narrow corner is kv-canary's only remaining originality. Source:
https://github.com/NVIDIA/kvpress · https://arxiv.org/abs/2512.12008

## Q3 — StreamingLLM re-roping (CONFIRMED BUG, 3-0)

StreamingLLM **requires** caching keys *pre*-RoPE and assigning positions by **cache index**, not
original token position. Paper (`arXiv:2309.17453`, ICLR 2024):
> "When determining the relative distance and adding positional information to tokens, StreamingLLM
> focuses on positions within the cache rather than those in the original text." … "For encoding like
> RoPE, we cache the Keys of tokens prior to introducing the rotary transformation."

Worked example: cache `[0,1,2,3,6,7,8]` decoding token 9 uses positions `[0,1,2,3,4,5,6,7]`.
HF transformers issue #35350 confirms post-RoPE eviction yields "confused position"; KVPress ships a
`KeyRerotationPress` to correct it. **kv-canary evicts from a post-RoPE HuggingFace DynamicCache
without re-roping → degraded-for-the-wrong-reason.** Source: https://arxiv.org/abs/2309.17453 ·
https://github.com/huggingface/transformers/issues/35350 · https://github.com/NVIDIA/kvpress

## Q4 — SnapKV per-head + pooling (CONFIRMED MISREPRESENTATION, 3-0)

SnapKV (`arXiv:2404.14469`, NeurIPS 2024) selects "clustered important KV positions **for each
attention head**", preserving the head dimension throughout, and a **1D max-pooling clustering step
is mandatory** (ablated as essential: "with the pooling … performs significantly better"). Observation
window = last segment of the prompt (~16–64 tokens) used to vote; top-k per head after pooling.
**kv-canary averages attention over all heads into one per-layer keep set with no pooling — that is
not SnapKV.** Source: https://arxiv.org/abs/2404.14469 · https://github.com/FasterDecoding/SnapKV

## Q5 — Perplexity protocol & quant memory (medium confidence)

- A single **0.5 context/continuation split** is legitimate but **non-standard**; the HF/NLP
  convention is **sliding-window (strided) perplexity**. A single split underestimates variance and
  is cherry-pickable. (https://huggingface.co/docs/transformers/perplexity)
- **`kv_bytes_retained = bits/16`** ignores per-tensor/group **scale + zero-point** overhead, so true
  retained memory is higher than reported — compression is modestly **overstated**. Per-tensor absmax
  quant is also cruder than the per-token/per-channel group-wise quant used in real KV-quant (KIVI,
  KVQuant), so kv-canary's quant results are likely **pessimistic**.

## Deeper problems (matter more than the bugs)

1. **Design vs. failure mode.** VeriCache's functional failures arise during **long decode**;
   kv-canary compresses the **prompt once, no decode-time rolling**. `2512.12008` finds decode-time
   compression is what governs reasoning correctness. The prompt-only design may be **confounded for
   the very code/tool-over-long-decode failures it targets.**
2. **"Why not just use KVPress?"** Fixing Q3+Q4 properly ≈ reimplementing KVPress (worse). NVIDIA
   maintains it.

## Recommended direction

Reframe the contribution as **executable / structured-output correctness (code-exec pass@k +
JSON/tool-call schema validity) under *decode-time* KV compression**, ideally **built on top of
NVIDIA KVPress** rather than a solo reimplementation — inheriting KVPress's correct StreamingLLM/
SnapKV (Q3/Q4 gone), targeting the one gap KVPress's accuracy CLI doesn't cover, and landing in a
high-visibility NVIDIA repo. If kept standalone: drop the broad wedge, frame "perplexity warns?" as a
hypothesis, fix re-roping + per-head-SnapKV-with-pooling, and add decode-time compression.

## Open questions

- Is there ANY existing tool scoring executable-code + tool-schema conformance under KV compression?
  If not, that narrow wedge is the whole contribution.
- Authoritative magnitudes for Q5 (strided-perplexity doc; KIVI/KVQuant overhead bytes).
- Can a prompt-only design generalize to decode-time functional failures at all?

## Caveats on this audit

Fast-moving subfield (key papers Oct/Dec 2025, May 2026; KVPress v0.3.0 June 2026) — the prior-art
picture can shift in months. Q1–Q4 rest on primary sources with unanimous/near-unanimous votes and
are robust; Q5 is medium confidence (lives inside a synthesizing verifier's note, not a separately
primary-cited claim).
