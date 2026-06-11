from kvcanary.types import Sample, EvalResult, CellSpec, CellResult


def test_sample_and_results_roundtrip():
    s = Sample(id="p1", prompt="hi", meta={"k": 1})
    assert s.id == "p1" and s.meta["k"] == 1
    e = EvalResult(scores={"pass@1": 1.0}, detail={"stderr": ""})
    assert e.scores["pass@1"] == 1.0
    spec = CellSpec(model="m", method="full", budget=1.0, task="code")
    assert spec.cell_id == "m|full|1.0|code"
    row = CellResult(
        spec=spec,
        sample_id="p1",
        scores={"pass@1": 1.0},
        raw_output="x",
        n_tokens=3,
        latency_s=0.1,
        kv_bytes_retained=1.0,
    )
    d = row.to_row()
    assert (
        d["model"] == "m"
        and d["scores"]["pass@1"] == 1.0
        and d["sample_id"] == "p1"
    )


def test_default_dicts_are_independent_instances():
    a = Sample(id="a", prompt="x")
    b = Sample(id="b", prompt="y")
    assert a.meta is not b.meta            # default_factory gives each its own dict
    a.meta["k"] = 1
    assert b.meta == {}
