from kvcanary.tasks.code_exec import CodeExecTask, run_in_sandbox

GOOD = "def add(a, b):\n    return a + b\n"
BAD  = "def add(a, b):\n    return a - b\n"
CHECK = "assert add(2, 3) == 5\n"

def test_sandbox_pass_fail_and_timeout():
    assert run_in_sandbox(GOOD + CHECK, timeout=5).ok is True
    assert run_in_sandbox(BAD + CHECK, timeout=5).ok is False
    assert run_in_sandbox("while True: pass\n", timeout=2).ok is False  # timed out

def test_code_task_scores_pass_at_1():
    t = CodeExecTask(problems=[{"id": "add", "prompt": "write add", "check": CHECK}])
    s = t.build_samples()[0]
    assert t.evaluate(s, output=GOOD).scores["pass@1"] == 1.0
    assert t.evaluate(s, output=BAD).scores["pass@1"] == 0.0

def test_code_task_ignores_extra_kwargs():
    t = CodeExecTask(problems=[{"id": "add", "prompt": "write add", "check": CHECK}])
    s = t.build_samples()[0]
    assert t.evaluate(s, output=GOOD, ppl=99.0).scores["pass@1"] == 1.0
