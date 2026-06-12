import pytest

from kvcanary.tasks.loaders import (
    load_code_problems,
    load_ppl_texts,
    load_tool_problems,
)
from kvcanary.tasks.tool_call import ToolCallTask
from kvcanary.tasks.code_exec import CodeExecTask


# ----- offline (builtin) loaders: schema + wiring, no network -----

def test_builtin_loaders_respect_limit_and_schema():
    code = load_code_problems(2, source="builtin")
    assert len(code) == 2
    assert all({"id", "prompt", "prefix", "check"} <= p.keys() for p in code)

    ppl = load_ppl_texts(2, source="builtin")
    assert len(ppl) == 2 and all(isinstance(t, str) for t in ppl)

    tool = load_tool_problems(5, source="builtin")
    assert len(tool) == 5


def test_tool_set_matches_gold_schema_and_evaluates():
    problems = load_tool_problems(100, source="builtin")   # all of them
    assert len(problems) >= 10
    task = ToolCallTask(problems)
    samples = task.build_samples()
    for s, p in zip(samples, problems):
        gold = s.meta["gold"]
        assert set(gold) == {"name", "arguments"} and isinstance(gold["arguments"], dict)
        # a model emitting the gold JSON verbatim must score a perfect call
        import json
        res = task.evaluate(s, output=json.dumps(p["gold"]))
        assert res.scores == {"json_valid": 1.0, "func_correct": 1.0, "arg_match": 1.0}


def test_code_prefix_forms_runnable_program():
    # Simulates HumanEval: signature+docstring in `prefix`, model "completes" the body.
    task = CodeExecTask([{
        "id": "sq", "prompt": "complete it",
        "prefix": "def square(x):\n    '''return x*x'''\n",
        "check": "assert square(5) == 25\n",
    }])
    sample = task.build_samples()[0]
    passing = task.evaluate(sample, output="    return x * x\n")
    failing = task.evaluate(sample, output="    return x + x\n")
    assert passing.scores["pass@1"] == 1.0
    assert failing.scores["pass@1"] == 0.0


# ----- real benchmarks: only when explicitly downloading (kept out of default CI) -----

@pytest.mark.network
def test_real_humaneval_and_wikitext_load():
    pytest.importorskip("datasets")
    code = load_code_problems(3, source="real")
    assert len(code) == 3
    assert all(p["id"].startswith("HumanEval/") for p in code)
    assert all("check(" in p["check"] for p in code)        # entry_point call appended

    ppl = load_ppl_texts(3, source="real")
    assert len(ppl) == 3 and all(len(t) > 100 for t in ppl)
