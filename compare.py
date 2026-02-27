import csv

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

x16 = read_ncu_csv("profile_x16.csv")
x32 = read_ncu_csv("profile_x32.csv")

diffs = []
for k in x16:
    if k in x32:
        v16 = x16[k]
        v32 = x32[k]
        if isinstance(v16, float) and isinstance(v32, float):
            if v16 != 0:
                rel_diff = (v32 - v16) / v16
            elif v32 != 0:
                rel_diff = float('inf')
            else:
                rel_diff = 0.0
            
            abs_diff = v32 - v16
            
            if abs(rel_diff) > 0.05 and abs(abs_diff) > 10:
                diffs.append((k, v16, v32, rel_diff, abs_diff))

diffs.sort(key=lambda x: abs(x[3]), reverse=True)

print(f"{'Metric':<80} | {'x16':<15} | {'x32':<15} | {'Rel Diff':<10}")
print("-" * 130)
for k, v16, v32, rel, abs_d in diffs[:50]:
    print(f"{k:<80} | {v16:<15.2f} | {v32:<15.2f} | {rel*100:>8.2f}%")
