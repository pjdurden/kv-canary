from kvcanary.metrics.aggregate import aggregate_rows
from kvcanary.metrics.sds import silent_degradation_score

ROWS = [
    {"method": "full", "budget": 1.0, "task": "tool", "scores": {"func_correct": 1.0}},
    {"method": "full", "budget": 1.0, "task": "tool", "scores": {"func_correct": 1.0}},
    {"method": "snapkv", "budget": 0.25, "task": "tool", "scores": {"func_correct": 0.5}},
    {"method": "snapkv", "budget": 0.25, "task": "tool", "scores": {"func_correct": 0.7}},
]

def test_aggregate_means_per_cell():
    agg = aggregate_rows(ROWS)
    assert agg[("full", 1.0, "tool")]["func_correct"] == 1.0
    assert abs(agg[("snapkv", 0.25, "tool")]["func_correct"] - 0.6) < 1e-9

def test_sds_is_high_when_functional_drops_but_ppl_flat():
    # functional drops 40%, perplexity rises 2% -> SDS ~ 20
    sds = silent_degradation_score(func_full=1.0, func_comp=0.6,
                                   ppl_full=10.0, ppl_comp=10.2, eps=1e-3)
    assert sds > 15
    # graceful case: both drop together -> SDS ~ 1
    sds2 = silent_degradation_score(func_full=1.0, func_comp=0.6,
                                    ppl_full=10.0, ppl_comp=14.0, eps=1e-3)
    assert sds2 < 1.5

def test_sds_perplexity_improving_is_maximally_silent_not_graceful():
    # perplexity IMPROVES (10 -> 8) while functional accuracy drops 40%.
    # This is the most deceptive case -> SDS must be very high, NOT ~1.
    sds = silent_degradation_score(func_full=1.0, func_comp=0.6,
                                   ppl_full=10.0, ppl_comp=8.0, eps=1e-3)
    assert sds > 100

def test_sds_both_flat_is_zero():
    # no functional drop and no perplexity change -> no silent degradation
    sds = silent_degradation_score(func_full=1.0, func_comp=1.0,
                                   ppl_full=10.0, ppl_comp=10.0, eps=1e-3)
    assert sds == 0.0

def test_aggregate_handles_heterogeneous_keys_and_empty():
    assert aggregate_rows([]) == {}
    rows = [
        {"method": "snapkv", "budget": 0.5, "task": "tool", "scores": {"json_valid": 1.0, "arg_match": 0.0}},
        {"method": "snapkv", "budget": 0.5, "task": "tool", "scores": {"json_valid": 0.0, "arg_match": 0.0}},
    ]
    agg = aggregate_rows(rows)
    cell = agg[("snapkv", 0.5, "tool")]
    assert cell["json_valid"] == 0.5 and cell["arg_match"] == 0.0
