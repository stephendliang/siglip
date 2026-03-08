# B200 Session Playbook

## Quick start

```bash
./tools/b200_session.sh              # full session (~45 min)
./tools/b200_session.sh --phase 3    # rerun single phase (e.g., ncu only)
```

Outputs go to `data/session_YYYYMMDD_HHMMSS/` with machine info (nvidia-smi, clocks, git rev).

```bash
python3 tools/analyze_session.py data/session_*/   # uses latest session dir
```

Paste the summary into Claude Code for interpretation.

## Phases

| Phase | What | Time est. |
|-|-|-|
| 1 | Machine snapshot (nvidia-smi, clocks, git rev) | <1 min |
| 2 | `compare_all.py --runs 20 --grid-search` (builds, benchmarks, ANOVA) | 25-30 min |
| 3 | ncu source counters (all 3 kernels) + full profile | 5-10 min |
| 4 | cuBLAS SASS capture (JIT cache dump) | 5 min |

## Manual follow-ups

```bash
# Analyze source counters
python3 tools/analyze_source_counters.py data/session_*/source_counters_siglip_vision.csv

# SASS analysis of cuBLAS captures
python3 tools/sass_analysis.py data/session_*/cublas_sass_*.txt

# Calibration microbenchmarks (if needed)
make calibration && ./calibration > data/session_*/cal_output.txt
cuobjdump --dump-sass calibration > data/session_*/cal_sass.txt
python3 tools/sass_analysis.py data/session_*/cal_sass.txt --calibrate-compare --runtime data/session_*/cal_output.txt
```
