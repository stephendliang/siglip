# Token-efficient CUDA tool usage

Large kernel files and profiler dumps blow through the context budget fast.
These are the default tactics for tool-heavy sessions in this repo.

Baseline measurements (2026-04-22, o200k_base via `token_count.py`):

| file           | bytes  | lines | tokens |
|---              |---     |---    |---     |
| `fc2_w3.cu`    | 180 KB | 4117  | 56,651 |
| `fc2_w3x.cu`   | 31 KB  |  757  | 10,144 |
| `kernel_common.cuh`, `kernel_body.cuh` | — | — | (also large — measure before reading) |

A single full `cuobjdump --dump-sass` or `ncu --set full` can exceed any of
the above. Don't paste those into context.

## Tactics

- [grep_before_read.md](grep_before_read.md) — locate with grep, narrow Read with offset/limit
- [ifdef_branch_filter.md](ifdef_branch_filter.md) — preprocess away dead `#ifdef` branches
- [sass_symbol_only.md](sass_symbol_only.md) — symbols first, full SASS on disk only
- [ncu_filtered_metrics.md](ncu_filtered_metrics.md) — always `--metrics --csv`, never `--set full`
- [kernel_structural_summary.md](kernel_structural_summary.md) — cache a function/section map
- [bench_tail_only.md](bench_tail_only.md) — bench result is ~3 lines; `tail` it
- [git_over_reread.md](git_over_reread.md) — `git log`/`git show` beats fresh full reads
- [delegate_to_explore.md](delegate_to_explore.md) — spawn Explore for multi-file searches

## How to use these notes

The CLAUDE.md "Context efficiency" section points here. Before tool-heavy
work, skim the relevant tactic. When you discover a new one, add a note here
and a one-liner in CLAUDE.md.
