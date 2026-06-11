import os
from kvcanary.report.chart import render_divergence_chart, divergence_series
from kvcanary.report.markdown import render_results_md

# Realistic agg: functional and perplexity live in SEPARATE task cells per (method, budget).
AGG = {
    ("full", 1.0, "tool"): {"func_correct": 1.0},
    ("full", 1.0, "perplexity"): {"perplexity": 10.0},
    ("snapkv", 0.5, "tool"): {"func_correct": 0.9},
    ("snapkv", 0.5, "perplexity"): {"perplexity": 10.05},
    ("snapkv", 0.25, "tool"): {"func_correct": 0.6},
    ("snapkv", 0.25, "perplexity"): {"perplexity": 10.2},
}


def test_chart_writes_nonempty_png(tmp_path):
    out = tmp_path / "div.png"
    render_divergence_chart(AGG, str(out))
    assert os.path.exists(out) and out.stat().st_size > 0


def test_divergence_series_separates_and_normalizes():
    s = divergence_series(AGG)
    assert "full" not in s                       # baseline excluded from method lines
    snap = s["snapkv"]
    i = snap["budgets"].index(0.25)
    assert abs(snap["func_norm"][i] - 0.6) < 1e-9            # 0.6 / 1.0
    assert abs(snap["ppl_norm"][i] - 10.0 / 10.2) < 1e-6     # perplexity-quality ~0.98
    assert snap["func_norm"][i] < snap["ppl_norm"][i]        # the divergence


def test_markdown_joins_func_and_perplexity_for_sds():
    md = render_results_md(AGG)
    assert "snapkv" in md and "SDS" in md
    assert "20.0" in md          # snapkv@0.25: rel func drop .4 / rel ppl rise .02 = 20.0
    assert "0.25" in md


def test_markdown_sds_dash_when_no_perplexity():
    agg = {("full", 1.0, "code"): {"pass@1": 0.9},
           ("snapkv", 0.25, "code"): {"pass@1": 0.5}}
    md = render_results_md(agg)
    assert "-" in md             # SDS undefined without a perplexity baseline
    assert "0.500" in md         # functional still reported
