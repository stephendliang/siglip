# Bench logs: tail, not full

Bench binaries print a lot of setup (dims, pointers, bar status, iteration
times). The result is 1–5 lines at the end.

## Single binary

    ./fc2-w3-dgswizzle | tail -5

## Comprehensive sweep

    ./tools/bench.sh --comprehensive | tail -40

## Pluck the fused-ms column across runs

    grep -HE '^fused' data/bench_*/summary.txt | sort -k2 -n

## Storing for later

If a full log might matter later, redirect to disk and `tail` into context:

    ./fc2-w3-dgswizzle > /tmp/run.log 2>&1
    tail -10 /tmp/run.log

Read the full log off-disk if the tail suggests a real problem; don't
default to it.
