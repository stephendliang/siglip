## fc2_w3x — top-tier dispatch ncu (locked 1800 MHz, --clock-control none)

Follow-up to a user-supplied 16-variant wall sweep on fc2_w3x (clocks
~1834 MHz, unlocked DVFS). The sweep showed 8 variants clustered within
1.7 µs of each other at the top; this run takes the top 6 (the actual
fast cluster) into ncu under locked clocks for mechanism diagnosis.

User's wall sweep (n=10 each, ms_med, machine without ncu):

  variant         ms_med    ms_p5    ms_p95    ms_sd
  checkered       1.0047    1.0041   1.0065    0.0006
  baseline/dg8    1.0049    1.0045   1.0054    0.0003
  dgsnake         1.0050    1.0047   1.0068    0.0006
  dg4             1.0057    1.0051   1.0063    0.0003
  hilbert         1.0061    1.0058   1.0079    0.0006
  innerT          1.0063    1.0060   1.0082    0.0006
  zigzag          1.0063    1.0058   1.0065    0.0002
  rowmajor        1.0064    1.0061   1.0087    0.0007
  --- gap ---
  nlock           1.0568    1.0561   1.0627    0.0021
  nflat           1.1062    1.1048   1.1075    0.0009
  ncyrot          1.1068    1.1047   1.1084    0.0011
  nsnake          1.1457    1.1433   1.1484    0.0015
  ncycle          1.1904    1.1877   1.1919    0.0013

Top 8 are within ~1.7 µs of each other; SDs of 0.0002–0.0007 ms put many
of these inside each other's noise envelopes. Bottom 5 are real losers
(structural — synchronous wavefront / load imbalance).

### NCU comparison of the top 6 (single launch under NCU_PROFILE)

  variant     wall(ms)  cyc.max   cyc.avg   tens%   inst(M)  long_sb  short_sb  barrier  wait    DRAM rd  DRAM%   L2 sect  L2 hit
  ----------  --------  --------  --------  ------  -------  -------  --------  -------  ------  -------  ------  -------  ------
  checkered   1.0047    1905868   1886721   97.92   139.11   6.810    0.240     0.700    1.550   2.996GB  52.34   596.7M   67.02
  dgsw        1.0049    1908083   1888894   97.90   140.15   6.710    0.270     0.680    1.550   2.988GB  52.20   599.8M   67.44
  dgsnake     1.0050    1903568   1885520   97.96   140.26   6.700    0.280     0.680    1.550   2.971GB  52.10   599.9M   67.38
  dg4         1.0057    1908010   1889548   97.78   140.15   6.720    0.270     0.680    1.550   3.018GB  52.62   635.7M   65.11
  hilbert     1.0061    1913023   1894192   97.46   140.41   6.640    0.250     0.740    1.570   3.026GB  52.57   648.0M   66.52
  zigzag      1.0063    1900267   1883289   97.73   136.54   7.090    0.240     0.710    1.510   2.957GB  52.00   637.2M   68.73

DRAM amp (theoretical bias-only read = 2.852 GB):
  dgsw       1.0477×    checkered  1.0504×    dgsnake  1.0416×
  hilbert    1.0612×    zigzag     1.0369×    dg4      1.0581×

### Reads

**Top 3 (checkered, dgsw, dgsnake) are statistically indistinguishable.**
Wall spread = 0.3 µs over per-variant SDs of 0.6 µs. Anything < ~1 µs at
this precision is noise.

**Checkered's apparent 0.2 µs win contradicts its ncu metrics.** Higher
long_sb than dgsw (6.81 vs 6.71) AND higher DRAM amp (1.0504 vs 1.0477)
should make it slower, not faster. Most likely: noise. Do NOT switch
default off dgsw on the basis of this run.

**dgsnake is the only top-tier variant whose ncu metrics ALSO line up.**
Lowest long_sb of the cluster (tied with hilbert, but smaller spread —
6.700 vs hilbert's 6.640 with much higher amp). Lowest DRAM amp of the
top group (1.0416×). LOWEST cyc.max (1,903,568) of any variant in this
sweep. Yet wall is +1 µs vs dgsw — suggests there's some final-stage cost
not captured in this metric set, or the wall-cyc translation under
unlocked DVFS doesn't quite line up with locked-1800 ncu.

**zigzag wins on both DRAM amp (1.0369×, best in sweep) AND L2 hit rate
(68.73%, also best)** — but loses on long_sb (7.090, worst in top tier).
Net: +1.4 µs slower. Same pattern as the rowmajor finding from
20260423_224615 (better DRAM efficiency loses to long_sb).

**hilbert wins on long_sb (6.640, best in sweep)** but loses on DRAM amp
(1.0612×, worst). Net: +1.2 µs. Confirms long_sb and DRAM amp are
genuinely independent axes — hilbert's space-filling curve gives best
arrival pattern but worst working-set efficiency.

**dg4** (G=4 instead of G=8) is +0.8 µs vs dg8. Smaller group → less L2
working set per group, BUT also less cooperative A-loading (4 clusters
per group instead of 8). L2 hit rate craters (67.44 → 65.11) without a
long_sb improvement (6.71 → 6.72). Group size 8 remains the right G.

### Conclusion

dgswizzle (G=8, default) is still the right ship default. The 16-variant
wall sweep shows it tied with checkered and dgsnake at sub-µs precision,
but ncu metrics give checkered no mechanistic credibility (it should be
slower) and dgsnake only marginal credibility (lowest amp + cyc.max but
+1 µs wall). Real losers (nlock/nflat/ncyrot/nsnake/ncycle) all match
prior structural predictions: synchronous N-wavefront, load imbalance.

The clock difference between user's sweep (~1834 MHz, unlocked DVFS) and
this ncu run (1800 MHz, locked) introduces a ~1.9% multiplier — user's
1.0049 ms ≈ 1.024 ms at locked 1800, vs my prior locked-1800 dgsw bench
of 1.034 ms (~1% gap, within run-to-run noise across machines).

### Files
  dgsw.csv checkered.csv dgsnake.csv hilbert.csv zigzag.csv dg4.csv
  *.stdout *.stderr  — program output / ncu warnings
