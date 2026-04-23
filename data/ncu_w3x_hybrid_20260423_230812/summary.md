## fc2_w3x — 3 hybrid dispatches vs dgswizzle (locked 1800 MHz, NCU)

Three hybrid dispatches added to tile_dispatch.cuh (TILE_DISPATCH=30/31/32),
each motivated by the question "can we mix dgswizzle + rowmajor (or any
secondary) to get the best of both worlds?"

All three are bijections on the FC2 tile space (10878 tiles), Python-verified
before build (see /tmp/bijection_check.py-equivalent inline check). All four
binaries (dgsw + 3 hybrids) are PACKED_TILES, NS=6, 6-warp persistent — only
the tile-order function differs. PASS errors=0/32 on every run.

### Wall (host bench, real packed FP8 inputs, 3 reps each)

| variant | dispatch                       | wall (ms) | TFLOPS | Δ vs dgsw |
|---------|--------------------------------|-----------|--------|-----------|
| dgsw    | dgswizzle (G=8, baseline)      | **1.034** | 4237   | —         |
| chet    | cluster-heterogeneous (TD=30)  | 1.035     | 4230   | +1 µs     |
| pmix    | phase-mixed (TD=31)            | 1.215     | 3605   | **+181 µs** |
| ingh    | within-group hybrid (TD=32)    | 1.037     | 4225   | +3 µs     |

### Hybrid designs

**CHET (TD=30, cluster-heterogeneous)**
- First 37 clusters: dgswizzle on M-blocks [0, 1813)
- Last 37 clusters: rowmajor on M-blocks [1813, 3626)
- Bijection by M-half partition; both halves disjoint, fully covered.

**PMIX (TD=31, phase-mixed)**
- Every cluster: first 98 tt's = dgswizzle on tn∈{0,1} (covers 7252 tiles)
- Every cluster: last 49 tt's = rowmajor on tn=2 (covers 3626 tiles)
- Bijection by N-column partition.

**INGH (TD=32, within-group hybrid)**
- Same dgswizzle group structure (G=8 → 24 tiles per group)
- Even group_idx: M-major in-group traversal (dgswizzle default)
- Odd group_idx:  N-major in-group traversal (DG_INNER_T default)
- Bijection per-group; disjoint groups compose.

### NCU comparison (single launch under NCU_PROFILE, --clock-control none)

| metric         |    dgsw |    chet |    pmix |    ingh |
|----------------|--------:|--------:|--------:|--------:|
| sm_cyc.max     | 1906145 | 1910523 | 2185199 | 1905049 |
| sm_cyc.avg     | 1886203 | 1889648 | 2169921 | 1886003 |
| tensor%        |   97.99 |   97.92 |   84.42 |   97.85 |
| inst (M)       |  140.15 |  138.45 |  138.67 |  138.00 |
| **long_sb**    |  **6.70** | **6.86** | **8.51** | **6.93** |
| short_sb       |    0.27 |    0.25 |    0.25 |    0.26 |
| barrier        |    0.68 |    0.70 |    0.71 |    0.70 |
| wait           |    1.55 |    1.53 |    1.56 |    1.51 |
| DRAM rd (GB)   |   2.990 |   3.008 |  *5.724* |   3.014 |
| DRAM wr (GB)   |   1.403 |   1.404 |   1.417 |   1.405 |
| DRAM%          |   52.31 |   52.42 |   74.73 |   52.58 |
| L2 sect (M)    |   602.1 |   603.9 |   654.6 |   623.3 |
| L2 hit%        |   67.27 |   68.28 |   51.82 |   69.13 |

DRAM amp (theoretical bias-only read = A + B + bias = 2.852 GB):

| variant | actual rd | amp     |
|---------|-----------|---------|
| dgsw    | 2.990 GB  | 1.048×  |
| chet    | 3.008 GB  | 1.055×  |
| pmix    | 5.724 GB  | **2.007×** (!) |
| ingh    | 3.014 GB  | 1.057×  |

### Verdict per variant

**CHET (+1 µs, ~wash).** The M-half partition averages dgswizzle's A-sharing
win on its half against rowmajor's tighter wavefront on the other. Half the
clusters keep dgswizzle's cooperative 8-cluster groups (just over a smaller
M-range); the other half do rowmajor. Net: long_sb 6.70 → 6.86, +1 µs wall.
The "best of both worlds" sums to the worst of each: dgswizzle's cluster
co-location works WITHIN its half but the two halves' wavefronts no longer
share L2 lines as effectively.

**PMIX (+181 µs, catastrophic).** Phase 2 is the killer: 49 consecutive tt's
where all 74 clusters work on tn=2 only. This is essentially the ncycle
pattern (CLAUDE.md: 1.226 ms) re-invented. Two failure modes compound:
1. **L2 collapse on phase transition**: phase 1 fills L2 with tn=0,1 working
   set; phase 2 abandons that for fresh tn=2 reads. L2 hit rate drops
   67.27% → 51.82% (−15 pts). DRAM amp doubles to 2.007×.
2. **Synchronous wavefront in phase 2**: 74 clusters synchronized on the
   same tn means their B-loads + A-loads land in lockstep, long_sb spikes
   from 6.70 → 8.51, tensor pipe drops from 97.99% → 84.42%.
Phase-mixing different rules works only if the rules' working sets compose.
N-column partition does not.

**INGH (+3 µs).** Alternating in-group traversal between M-major and N-major
preserves dgswizzle's cross-cluster A-sharing at group level (same 8 M-blocks
in play across 24 cluster slots per group), so DRAM amp barely budges
(1.048× → 1.057×). long_sb up by 0.23 (6.70 → 6.93). The within-group
permutation breaks the per-cluster TN-run pattern (cluster c=0 used to do
tn=0,0,0,...,1,1,1,...,2,2 in a row across many groups; now alternates per
group). That tiny break costs 3 µs. Same magnitude as plain rowmajor — same
mechanism (TN-cycling raises long_sb).

### Cross-cutting observation

All three hybrids LOSE, by amounts predictable from the long_scoreboard
delta vs dgsw:

| variant | Δ long_sb | Δ wall (µs) |
|---------|-----------|-------------|
| chet    | +0.16     | +1          |
| ingh    | +0.23     | +3          |
| pmix    | +1.81     | +181        |

Each +0.1 in long_sb costs ~1 µs of wall on this kernel — a tighter
correlation than the prior dispatch sweep showed. The "structural
correlate" thesis from CLAUDE.md (long_sb arrival-pattern is the
critical-path bottleneck, not DRAM amp) is reinforced: even a HYBRID that
keeps dgswizzle's cooperative A-sharing on half the clusters can't avoid a
small long_sb hit, and pays for it.

The data does NOT identify a hybrid that beats dgsw. dgswizzle's cooperative
8-cluster A-loading wavefront is structurally complete — splitting,
phase-mixing, or in-group permuting all break a load-bearing piece of it
without finding a compensating win elsewhere.

### Files
  dgsw.csv chet.csv pmix.csv ingh.csv  — raw ncu CSVs (23 metrics each)
  *.stdout *.stderr                    — program output / ncu warnings
