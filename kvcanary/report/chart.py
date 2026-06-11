import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _functional(scores: dict):
    """Functional accuracy from a functional-task cell, or None if this cell has none."""
    if "func_correct" in scores:
        return scores["func_correct"]
    if "pass@1" in scores:
        return scores["pass@1"]
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _join(agg: dict) -> dict:
    """Per method -> {budget -> {"func": [vals], "ppl": value|None}}, joining task cells."""
    data = {}
    for (method, budget, _task), s in agg.items():
        d = data.setdefault(method, {}).setdefault(budget, {"func": [], "ppl": None})
        f = _functional(s)
        if f is not None:
            d["func"].append(f)
        if "perplexity" in s:
            d["ppl"] = s["perplexity"]
    return data


def divergence_series(agg: dict) -> dict:
    """Per non-full method: budgets, functional normalized to full, perplexity-quality normalized to full.

    func_norm = func / full_func   (1.0 = no functional loss).
    ppl_norm  = full_ppl / ppl     (1.0 = no perplexity rise; <1 = perplexity got worse).
    The divergence finding = func_norm cliffs while ppl_norm hugs 1.0.
    """
    data = _join(agg)
    base = data.get("full", {}).get(1.0, {"func": [], "ppl": None})
    base_func = _mean(base["func"]) or 1.0
    base_ppl = base["ppl"] or 1.0
    out = {}
    for method, budgets in data.items():
        if method == "full":
            continue
        pts = sorted(budgets.items())
        out[method] = {
            "budgets": [b for b, _ in pts],
            "func_norm": [(_mean(d["func"]) / base_func) if d["func"] else float("nan")
                          for _, d in pts],
            "ppl_norm": [(base_ppl / d["ppl"]) if d["ppl"] else float("nan")
                         for _, d in pts],
        }
    return out


def render_divergence_chart(agg: dict, out_path: str):
    series = divergence_series(agg)
    fig, ax = plt.subplots()
    for method, s in series.items():
        line, = ax.plot(s["budgets"], s["func_norm"], marker="o",
                        label=f"{method} functional")
        ax.plot(s["budgets"], s["ppl_norm"], marker="x", linestyle="--",
                color=line.get_color(), label=f"{method} perplexity-quality")
    ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("KV memory retained (fraction, log scale)")
    ax.set_ylabel("score normalized to full (1.0 = no degradation)")
    ax.legend(fontsize="small")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
