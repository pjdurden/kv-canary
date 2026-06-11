import json
import os
import sys

from kvcanary.metrics.aggregate import aggregate_rows
from kvcanary.report.chart import render_divergence_chart
from kvcanary.report.markdown import render_results_md

with open(sys.argv[1]) as f:
    rows = [json.loads(line) for line in f if line.strip()]
agg = aggregate_rows(rows)
os.makedirs("report", exist_ok=True)
render_divergence_chart(agg, "report/divergence.png")
with open("report/RESULTS.md", "w") as f:
    f.write(render_results_md(agg))
print("wrote report/divergence.png and report/RESULTS.md")
