def silent_degradation_score(func_full, func_comp, ppl_full, ppl_comp, eps=1e-3) -> float:
    """Silent Degradation Score = (relative functional drop) / (relative perplexity rise).

    High SDS = functional accuracy craters while the token metric (perplexity) fails to warn.
    The most deceptive case is when perplexity stays flat OR IMPROVES (ppl_comp <= ppl_full)
    while functional accuracy drops: rel_ppl_drop clamps to 0, so SDS = rel_func_drop / eps,
    intentionally large. That is NOT graceful — perplexity improving while function breaks is the
    maximally silent failure this metric exists to surface. ("Graceful" is perplexity rising in
    step with the functional drop, giving SDS ~ 1.) The magnitude in the flat/improving regime is
    eps-bounded and not meaningful beyond "very high"; treat SDS as an ordinal silentness signal.
    Improvements clamp to 0 (a functional gain is not a drop).
    """
    rel_func_drop = max(0.0, (func_full - func_comp) / max(func_full, eps))
    rel_ppl_drop = max(0.0, (ppl_comp - ppl_full) / max(ppl_full, eps))
    return rel_func_drop / max(rel_ppl_drop, eps)
