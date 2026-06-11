from kvcanary.runner.config import load_cells


def test_load_cells_expands_matrix(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "models: [toy]\n"
        "tasks: [code, perplexity]\n"
        "compressors:\n"
        "  - {method: full, budget: 1.0}\n"
        "  - {method: streamingllm, budget: 0.5}\n"
    )
    cells = load_cells(str(y))
    ids = {c.cell_id for c in cells}
    assert "toy|full|1.0|code" in ids
    assert "toy|streamingllm|0.5|perplexity" in ids
    assert len(cells) == 1 * 2 * 2  # models x compressors x tasks
