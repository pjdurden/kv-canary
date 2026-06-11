from collections import defaultdict

def aggregate_rows(rows: list[dict]) -> dict:
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["method"], r["budget"], r["task"])
        for k, v in r["scores"].items():
            sums[key][k] += v
            counts[key][k] += 1
    return {key: {k: sums[key][k] / counts[key][k] for k in sums[key]} for key in sums}
