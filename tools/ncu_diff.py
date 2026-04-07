import csv
import sys

def read_ncu_csv(filepath):
    """Read ncu CSV (one row per metric) into {metric_name: (float_value, unit)}."""
    metrics = {}
    with open(filepath) as f:
        for row in csv.DictReader(f):
            name = row.get("Metric Name", "")
            val_str = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            if not name or val_str in ("n/a", "N/A", ""):
                continue
            try:
                metrics[name] = (float(val_str.replace(",", "")), unit)
            except ValueError:
                pass
    return metrics

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <baseline.csv> <experiment.csv>")
    sys.exit(1)

baseline = read_ncu_csv(sys.argv[1])
experiment = read_ncu_csv(sys.argv[2])

diffs = []
for k in baseline:
    if k in experiment:
        vb, ub = baseline[k]
        ve, ue = experiment[k]
        if vb != 0:
            rel_diff = (ve - vb) / vb
        elif ve != 0:
            rel_diff = float('inf')
        else:
            rel_diff = 0.0

        abs_diff = ve - vb
        if abs(rel_diff) > 0.05 and abs(abs_diff) > 10:
            diffs.append((k, vb, ve, rel_diff, abs_diff, ub))

diffs.sort(key=lambda x: abs(x[3]), reverse=True)

print(f"{'Metric':<70} | {'Baseline':<15} | {'Experiment':<15} | {'Rel Diff':<10} | {'Unit'}")
print("-" * 125)
for k, vb, ve, rel, abs_d, unit in diffs[:50]:
    def fmt(v):
        if v >= 1e6:
            return f"{v:>13,.0f}"
        elif v >= 100:
            return f"{v:>13,.1f}"
        else:
            return f"{v:>13.2f}"
    print(f"{k:<70} | {fmt(vb)} | {fmt(ve)} | {rel*100:>8.1f}% | {unit}")
