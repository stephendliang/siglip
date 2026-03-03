import csv
import sys

def read_ncu_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        units = next(reader)
        values = next(reader)

        metrics = {}
        for h, u, v in zip(headers, units, values):
            try:
                metrics[h] = float(v)
            except ValueError:
                metrics[h] = v
        return metrics

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <baseline.csv> <experiment.csv>")
    sys.exit(1)

baseline = read_ncu_csv(sys.argv[1])
experiment = read_ncu_csv(sys.argv[2])

diffs = []
for k in baseline:
    if k in experiment:
        vb = baseline[k]
        ve = experiment[k]
        if isinstance(vb, float) and isinstance(ve, float):
            if vb != 0:
                rel_diff = (ve - vb) / vb
            elif ve != 0:
                rel_diff = float('inf')
            else:
                rel_diff = 0.0

            abs_diff = ve - vb

            if abs(rel_diff) > 0.05 and abs(abs_diff) > 10:
                diffs.append((k, vb, ve, rel_diff, abs_diff))

diffs.sort(key=lambda x: abs(x[3]), reverse=True)

print(f"{'Metric':<80} | {'Baseline':<15} | {'Experiment':<15} | {'Rel Diff':<10}")
print("-" * 130)
for k, vb, ve, rel, abs_d in diffs[:50]:
    print(f"{k:<80} | {vb:<15.2f} | {ve:<15.2f} | {rel*100:>8.2f}%")
