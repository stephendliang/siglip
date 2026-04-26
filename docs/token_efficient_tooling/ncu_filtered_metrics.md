# ncu: targeted --metrics + CSV

Default `ncu` or `--set full` reports are huge and mostly irrelevant to any
targeted question. Always constrain metrics and use CSV.

## Rule

- **Never** `--set full` into context.
- **Always** specify `--metrics` + `--csv`.
- Pipe through `tools/ncu_anova.py` for comparisons; read the summary, not
  the raw tensor.

## Canonical narrow metric sets

Stall arrival-pattern (the `long_scoreboard` investigation):

    --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum,\
    smsp__pcsamp_warps_issue_stalled_barrier.sum,\
    smsp__pcsamp_warps_issue_stalled_wait.sum,\
    smsp__pcsamp_warps_issue_stalled_mio_throttle.sum

DRAM / L2:

    --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct,\
    l1tex__t_bytes.sum

TMA / store path:

    --metrics smsp__inst_executed_pipe_tma.sum,\
    smsp__inst_executed_pipe_tensor_op_hmma.sum

## Store results to disk

    ncu --csv --metrics ... ./fc2-w3-dgswizzle > data/ncu_$(date +%Y%m%d_%H%M%S).csv

Read the CSV into your summary tool, not into Claude's context.
