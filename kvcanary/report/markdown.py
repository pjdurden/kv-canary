from kvcanary.metrics.sds import silent_degradation_score


def _functional(scores: dict):
    if "func_correct" in scores:
        return scores["func_correct"]
    if "pass@1" in scores:
        return scores["pass@1"]
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def render_results_md(agg: dict) -> str:
    """Per (method, budget) table joining functional accuracy + perplexity, with SDS.

    SDS needs both a functional and a perplexity baseline; it renders as "-" when either is
    absent (e.g. a code-only run with no perplexity task).
    """
    joined = {}
    for (method, budget, _task), s in agg.items():
        d = joined.setdefault((method, budget), {"func": [], "ppl": None})
        f = _functional(s)
        if f is not None:
            d["func"].append(f)
        if "perplexity" in s:
            d["ppl"] = s["perplexity"]
    full = joined.get(("full", 1.0), {"func": [], "ppl": None})
    full_func = _mean(full["func"])
    full_ppl = full["ppl"]
    lines = ["| method | budget | functional | perplexity | SDS |",
             "|---|---|---|---|---|"]
    for (method, budget), d in sorted(joined.items()):
        func = _mean(d["func"])
        ppl = d["ppl"]
        func_s = f"{func:.3f}" if func is not None else "-"
        ppl_s = f"{ppl:.2f}" if ppl is not None else "-"
        if method == "full":
            sds_s = "0.0"
        elif None not in (func, full_func, ppl, full_ppl):
            sds_s = f"{silent_degradation_score(full_func, func, full_ppl, ppl):.1f}"
        else:
            sds_s = "-"
        lines.append(f"| {method} | {budget} | {func_s} | {ppl_s} | {sds_s} |")
    return "\n".join(lines) + "\n"
