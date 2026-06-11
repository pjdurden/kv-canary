from kvcanary.tasks.perplexity import PerplexityTask

def test_perplexity_task_uses_backend_score():
    t = PerplexityTask(texts=["the quick brown fox"])
    samples = t.build_samples()
    assert len(samples) == 1
    res = t.evaluate(samples[0], output="", ppl=23.0)
    assert res.scores["perplexity"] == 23.0
