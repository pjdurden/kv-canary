from kvcanary.tasks.tool_call import ToolCallTask

PROB = {
    "id": "weather", "prompt": "call get_weather for Paris",
    "gold": {"name": "get_weather", "arguments": {"city": "Paris"}},
}

def _task(): return ToolCallTask(problems=[PROB])

def test_valid_correct_call_scores_all_ones():
    s = _task().build_samples()[0]
    r = _task().evaluate(s, '{"name": "get_weather", "arguments": {"city": "Paris"}}')
    assert r.scores == {"json_valid": 1.0, "func_correct": 1.0, "arg_match": 1.0}

def test_malformed_json_fails_everything():
    s = _task().build_samples()[0]
    r = _task().evaluate(s, '{name: get_weather')           # not JSON
    assert r.scores["json_valid"] == 0.0 and r.scores["arg_match"] == 0.0

def test_wrong_function_and_wrong_arg():
    s = _task().build_samples()[0]
    r1 = _task().evaluate(s, '{"name": "get_news", "arguments": {"city": "Paris"}}')
    assert r1.scores["json_valid"] == 1.0 and r1.scores["func_correct"] == 0.0
    r2 = _task().evaluate(s, '{"name": "get_weather", "arguments": {"city": "London"}}')
    assert r2.scores["func_correct"] == 1.0 and r2.scores["arg_match"] == 0.0

def test_valid_json_missing_arguments_key():
    s = _task().build_samples()[0]
    r = _task().evaluate(s, '{"name": "get_weather"}')   # parses, right func, no args
    assert r.scores["json_valid"] == 1.0
    assert r.scores["func_correct"] == 1.0
    assert r.scores["arg_match"] == 0.0                  # .get("arguments") is None != gold args
