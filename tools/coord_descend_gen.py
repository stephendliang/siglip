#!/usr/bin/env python3
"""coord_descend_gen.py — emit gflip-family m-axis neighborhood templates.

Mines the parameter space around blkswap (TD=54) + lmrev (TD=52) — both
front-tier-tied at n=43910 — to map out where in the m-axis perturbation
neighborhood the wall optimum actually lies.  The bloom filter cannot
discriminate within this family from sliding-window L2-warmth proxies,
so we generate ~20 byte-cheap templates, bloom-rank them, build the
WORTHY ones, and let B200 wall pick the winner.

Grid (G=DGG=8 throughout):
  XK   ∈ {1, 2, 3, 5, 7}                        (pairing XOR-key on group_idx)
  p_u  ∈ {id, bitrev3, mul3mod8, mul5mod8}      (uniform within-group lm permutation)
  alt  ∈ {none, xor1..xor7, mul3, mul5, bitrev} (alt-group lm permutation)
  dens ∈ {all, alt0, alt1, qrt0..qrt3, n3k0..n3k2} (which groups get alt)

Already in tree (skipped — see fc2_w3x.cu TD=33/52/54/56/57/58):
  gflip, gflip_lmrev, gflip_blkswap, gflip_blklmrev, gflip_blkmul3, gflip_quartswap

Output (under tools/_coord_gen/):
  cuda_templates.cuh    — paste between markers in fc2_w3x.cu
  cuda_dispatch.frag    — switch arms for tile_swizzle_t<TD,DGG>
  cuda_table.frag       — VCFG entries for VARIANTS[]
  python_funcs.py.frag  — sw_<name> defs for analyze_swizzle.py
  python_table.frag     — entries for the VARIANTS list in analyze_swizzle.py
  sweep_names.txt       — comma-joined list for SWEEP=<...>

Workflow:
  1. python3 tools/coord_descend_gen.py             # validate + emit fragments
  2. splice into fc2_w3x.cu and tools/analyze_swizzle.py
  3. python3 tools/analyze_swizzle.py --csv data/coord_metrics.csv
  4. python3 tools/bloom_filter.py data/coord_metrics.csv
  5. SWEEP=<top WORTHY names> REPS=16384 tools/sweep_fc2_w3x_swizzle.sh

Cell selection:
  --grid default          # ~20 cells (XK probes + alt-XOR sweep + density + composition)
  --grid full             # full Cartesian product (~1300, validated only)
  --grid xk               # only XK probes
  --grid altxor           # only alt-group XOR sweep
  --grid density          # only density variants on xor4
  --grid composition      # uniform×alt compositions
"""
import argparse
import os
import sys

NIG = 8
TD_BASE = 80


def bitrev3(x):
    return ((x & 1) << 2) | (x & 2) | ((x >> 2) & 1)


P_UNIFORM = {
    "id":     lambda lm: lm,
    "bitrev": bitrev3,
    "mul3":   lambda lm: (lm * 3) & 7,
    "mul5":   lambda lm: (lm * 5) & 7,
}

P_ALT = {
    "none":   lambda lm: lm,
    "xor1":   lambda lm: lm ^ 1,
    "xor2":   lambda lm: lm ^ 2,
    "xor3":   lambda lm: lm ^ 3,
    "xor4":   lambda lm: lm ^ 4,
    "xor5":   lambda lm: lm ^ 5,
    "xor6":   lambda lm: lm ^ 6,
    "xor7":   lambda lm: lm ^ 7,
    "mul3":   lambda lm: (lm * 3) & 7,
    "mul5":   lambda lm: (lm * 5) & 7,
    "bitrev": bitrev3,
}

DENSITY = {
    "all":  lambda g: True,
    "alt0": lambda g: (g & 1) == 0,
    "alt1": lambda g: (g & 1) == 1,
    "qrt0": lambda g: (g & 3) == 0,
    "qrt1": lambda g: (g & 3) == 1,
    "qrt2": lambda g: (g & 3) == 2,
    "qrt3": lambda g: (g & 3) == 3,
    "n3k0": lambda g: (g % 3) == 0,
    "n3k1": lambda g: (g % 3) == 1,
    "n3k2": lambda g: (g % 3) == 2,
}

CUDA_UNIFORM = {
    "id":     None,
    "bitrev": "lm = ((lm_raw & 1) << 2) | (lm_raw & 2) | ((lm_raw >> 2) & 1);",
    "mul3":   "lm = (lm_raw * 3) & 7;",
    "mul5":   "lm = (lm_raw * 5) & 7;",
}

CUDA_ALT = {
    "xor1":   "lm = lm ^ 1;",
    "xor2":   "lm = lm ^ 2;",
    "xor3":   "lm = lm ^ 3;",
    "xor4":   "lm = lm ^ 4;",
    "xor5":   "lm = lm ^ 5;",
    "xor6":   "lm = lm ^ 6;",
    "xor7":   "lm = lm ^ 7;",
    "mul3":   "lm = (lm * 3) & 7;",
    "mul5":   "lm = (lm * 5) & 7;",
    "bitrev": "lm = ((lm & 1) << 2) | (lm & 2) | ((lm >> 2) & 1);",
}

CUDA_DENSITY = {
    "all":  "true",
    "alt0": "((group_idx & 1) == 0)",
    "alt1": "(group_idx & 1)",
    "qrt0": "((group_idx & 3) == 0)",
    "qrt1": "((group_idx & 3) == 1)",
    "qrt2": "((group_idx & 3) == 2)",
    "qrt3": "((group_idx & 3) == 3)",
    "n3k0": "((group_idx - (group_idx / 3) * 3) == 0)",
    "n3k1": "((group_idx - (group_idx / 3) * 3) == 1)",
    "n3k2": "((group_idx - (group_idx / 3) * 3) == 2)",
}

PY_UNIFORM = {
    "id":     None,
    "bitrev": "lm = ((lmr & 1) << 2) | (lmr & 2) | ((lmr >> 2) & 1)",
    "mul3":   "lm = (lmr * 3) % 8",
    "mul5":   "lm = (lmr * 5) % 8",
}

PY_ALT = {
    "xor1":   "lm = lm ^ 1",
    "xor2":   "lm = lm ^ 2",
    "xor3":   "lm = lm ^ 3",
    "xor4":   "lm = lm ^ 4",
    "xor5":   "lm = lm ^ 5",
    "xor6":   "lm = lm ^ 6",
    "xor7":   "lm = lm ^ 7",
    "mul3":   "lm = (lm * 3) % 8",
    "mul5":   "lm = (lm * 5) % 8",
    "bitrev": "lm = ((lm & 1) << 2) | (lm & 2) | ((lm >> 2) & 1)",
}

PY_DENSITY = {
    "all":  "True",
    "alt0": "(group_idx & 1) == 0",
    "alt1": "(group_idx & 1) == 1",
    "qrt0": "(group_idx & 3) == 0",
    "qrt1": "(group_idx & 3) == 1",
    "qrt2": "(group_idx & 3) == 2",
    "qrt3": "(group_idx & 3) == 3",
    "n3k0": "(group_idx % 3) == 0",
    "n3k1": "(group_idx % 3) == 1",
    "n3k2": "(group_idx % 3) == 2",
}

ALREADY_IN_TREE = {
    (1, "id",     "none",   "all"):  ("gflip",          33),
    (1, "bitrev", "none",   "all"):  ("gflip_lmrev",    52),
    (1, "id",     "xor4",   "alt1"): ("gflip_blkswap",  54),
    (1, "id",     "mul3",   "alt1"): ("gflip_blkmul3",  57),
    (1, "id",     "xor4",   "qrt1"): ("gflip_quartswap",58),
    (1, "bitrev", "xor4",   "alt1"): ("gflip_blklmrev", 56),
}


def check_bijection(xk, pu, alt, dens):
    """The (lm_raw → lm) map must be a permutation on [0,8) for every group_idx
    phase that the density predicate touches.  We sweep groups 0..23 to cover
    lcm(2,3,4) phases × 2."""
    pu_fn = P_UNIFORM[pu]
    alt_fn = P_ALT[alt]
    d_fn = DENSITY[dens]
    for g in range(24):
        out = set()
        for lm_raw in range(NIG):
            lm = pu_fn(lm_raw)
            if alt != "none" and d_fn(g):
                lm = alt_fn(lm)
            out.add(lm)
        if out != set(range(NIG)):
            return False, g, out
    return True, None, None


def emit_cuda(td, name, xk, pu, alt, dens, comment):
    pred = CUDA_DENSITY[dens]
    L = []
    L.append(f"/* TD={td}: {name}.  {comment}")
    L.append(f"            XK={xk}, p_u={pu}, alt={alt}, dens={dens}.  Bijection-checked. */")
    L.append("template<int DGG>")
    L.append("static __device__ __forceinline__")
    L.append(f"int {name}_swizzle_t(int lin) {{")
    L.append("    const int group_tiles = TILES_N * DGG;")
    L.append("    const int num_groups  = (TILES_M + DGG - 1) / DGG;")
    L.append("    int group_idx         = lin / group_tiles;")
    L.append("    const int in_group    = lin - group_idx * group_tiles;")
    if xk != 0:
        L.append(f"    const int paired      = group_idx ^ {xk};")
        L.append("    if (paired < num_groups) group_idx = paired;")
    L.append("    const int first_m = group_idx * DGG;")
    L.append("    const int nig     = (first_m + DGG <= TILES_M) ? DGG : TILES_M - first_m;")
    L.append("    const int lm_raw  = in_group - (in_group / nig) * nig;")
    L.append("    const int ln      = in_group / nig;")
    L.append("    int lm = lm_raw;")
    if pu != "id" or alt != "none":
        L.append("    if (nig == 8) {")
        if pu != "id":
            L.append(f"        {CUDA_UNIFORM[pu]}")
        if alt != "none":
            if pred == "true":
                L.append(f"        {CUDA_ALT[alt]}")
            else:
                L.append(f"        if ({pred}) {{ {CUDA_ALT[alt]} }}")
        L.append("    }")
    L.append("    int tm = first_m + lm;")
    L.append("    if (tm >= TILES_M) tm = TILES_M - 1;")
    L.append("    return tm * TILES_N + ln;")
    L.append("}")
    return "\n".join(L)


def emit_python(name, xk, pu, alt, dens, comment):
    pred = PY_DENSITY[dens]
    L = []
    L.append(f"def sw_{name}(lin, G=8):")
    L.append(f'    """{comment}  XK={xk}, p_u={pu}, alt={alt}, dens={dens}."""')
    L.append("    gt = TILES_N * G")
    L.append("    num_groups = (TILES_M + G - 1) // G")
    L.append("    group_idx = lin // gt")
    L.append("    in_g = lin - group_idx * gt")
    if xk != 0:
        L.append(f"    paired = group_idx ^ {xk}")
        L.append("    if paired < num_groups: group_idx = paired")
    L.append("    first_m = group_idx * G")
    L.append("    nig = G if first_m + G <= TILES_M else TILES_M - first_m")
    L.append("    lmr = in_g % nig")
    L.append("    ln = in_g // nig")
    L.append("    lm = lmr")
    if pu != "id" or alt != "none":
        L.append("    if nig == 8:")
        if pu != "id":
            L.append(f"        {PY_UNIFORM[pu]}")
        if alt != "none":
            if pred == "True":
                L.append(f"        {PY_ALT[alt]}")
            else:
                L.append(f"        if {pred}: {PY_ALT[alt]}")
    L.append("    return first_m + lm, ln")
    return "\n".join(L)


def default_cells():
    cells = []

    for xk in (2, 3, 5, 7):
        cells.append((xk, "id", "xor4", "alt1",
                      f"gflip_xk{xk}_blkswap",
                      f"XK={xk} pairing × blkswap-^4 alt1 (vs gflip's XK=1)"))

    for m in (1, 2, 3, 5, 6, 7):
        cells.append((1, "id", f"xor{m}", "alt1",
                      f"gflip_blkx{m}",
                      f"alt1 × xor{m} (vs blkswap's xor4) — alt-mask scan"))

    for dens in ("alt0", "qrt0", "qrt2", "qrt3", "n3k1"):
        cells.append((1, "id", "xor4", dens,
                      f"gflip_blk_{dens}",
                      f"xor4 × density={dens} (vs blkswap's alt1, quartswap's qrt1)"))

    for pu in ("mul3", "mul5"):
        cells.append((1, pu, "none", "all",
                      f"gflip_{pu}",
                      f"uniform p_u={pu} only (vs lmrev's bitrev)"))

    for pu, alt, dens in [
        ("bitrev", "xor1", "alt1"),
        ("bitrev", "xor2", "alt1"),
        ("mul3",   "xor4", "alt1"),
    ]:
        cells.append((1, pu, alt, dens,
                      f"gflip_{pu}_{alt}_{dens}",
                      f"composition: uniform={pu} + alt={alt} dens={dens}"))

    return cells


def all_cells():
    cells = []
    for xk in (1, 2, 3, 5, 7):
        for pu in P_UNIFORM:
            for alt in P_ALT:
                if alt == "none":
                    cells.append((xk, pu, alt, "all",
                                  f"gflip_xk{xk}_{pu}",
                                  f"all-cells: XK={xk} p_u={pu}"))
                    continue
                for dens in DENSITY:
                    if dens == "all" and alt == "none":
                        continue
                    name = f"gflip_xk{xk}_{pu}_{alt}_{dens}"
                    cells.append((xk, pu, alt, dens,
                                  name,
                                  f"all-cells: XK={xk} p_u={pu} alt={alt} dens={dens}"))
    return cells


def filter_grid(cells, mode):
    if mode == "default":
        return cells
    if mode == "full":
        return all_cells()
    if mode == "xk":
        return [c for c in cells if c[0] != 1]
    if mode == "altxor":
        return [c for c in cells if c[0] == 1 and c[1] == "id" and c[2].startswith("xor")
                and c[2] != "xor4" and c[3] == "alt1"]
    if mode == "density":
        return [c for c in cells if c[0] == 1 and c[1] == "id" and c[2] == "xor4"
                and c[3] not in ("alt1", "qrt1")]
    if mode == "composition":
        return [c for c in cells if c[1] != "id" and c[2] != "none"]
    raise SystemExit(f"unknown grid: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="default",
                    choices=("default", "full", "xk", "altxor", "density", "composition"))
    ap.add_argument("--out", default="tools/_coord_gen",
                    help="output directory for fragments")
    ap.add_argument("--td-base", type=int, default=TD_BASE,
                    help="first TD slot (default 80)")
    ap.add_argument("--validate-only", action="store_true",
                    help="check bijection and print the cell list, no file output")
    args = ap.parse_args()

    raw = default_cells() if args.grid == "default" else all_cells()
    cells = filter_grid(raw, args.grid) if args.grid not in ("default", "full") else raw

    valid, skipped_dup, skipped_nonbij = [], [], []
    for cell in cells:
        xk, pu, alt, dens, name, comment = cell
        key = (xk, pu, alt, dens)
        if key in ALREADY_IN_TREE:
            existing_name, existing_td = ALREADY_IN_TREE[key]
            skipped_dup.append((cell, existing_name, existing_td))
            continue
        ok, bad_g, bad_set = check_bijection(xk, pu, alt, dens)
        if not ok:
            skipped_nonbij.append((cell, bad_g, bad_set))
            continue
        valid.append(cell)

    print(f"=== coord_descend_gen: grid={args.grid} ===")
    print(f"  raw cells:        {len(cells)}")
    print(f"  already in tree:  {len(skipped_dup)}")
    print(f"  non-bijective:    {len(skipped_nonbij)}")
    print(f"  emit:             {len(valid)} (TD slots {args.td_base}..{args.td_base + len(valid) - 1})")
    if skipped_dup:
        print()
        print("  skipped (duplicate of in-tree variant):")
        for cell, en, etd in skipped_dup:
            print(f"    {cell[4]:30s} → already TD={etd} ({en})")
    if skipped_nonbij:
        print()
        print("  skipped (non-bijective at some group phase):")
        for cell, g, bs in skipped_nonbij:
            print(f"    {cell[4]:30s} fails at group_idx={g}: {sorted(bs)}")
    print()

    if args.validate_only:
        print("Cells that would be emitted:")
        for i, (xk, pu, alt, dens, name, comment) in enumerate(valid):
            td = args.td_base + i
            print(f"  TD={td:3d}  {name:32s}  XK={xk} p_u={pu:6s} alt={alt:6s} dens={dens}")
        return

    os.makedirs(args.out, exist_ok=True)

    cuda_path = os.path.join(args.out, "cuda_templates.cuh")
    with open(cuda_path, "w") as f:
        f.write("/* coord_descend_gen.py output — paste between BEGIN/END markers in fc2_w3x.cu */\n")
        f.write("/* BEGIN COORD_DESCEND */\n\n")
        for i, (xk, pu, alt, dens, name, comment) in enumerate(valid):
            td = args.td_base + i
            f.write(emit_cuda(td, name, xk, pu, alt, dens, comment))
            f.write("\n\n")
        f.write("/* END COORD_DESCEND */\n")

    disp_path = os.path.join(args.out, "cuda_dispatch.frag")
    with open(disp_path, "w") as f:
        f.write("/* tile_swizzle_t<TD,DGG> switch arms — splice into the if/else chain */\n")
        for i, (xk, pu, alt, dens, name, comment) in enumerate(valid):
            td = args.td_base + i
            f.write(f"    else if constexpr (TD == {td}) return {name}_swizzle_t<DGG>(lin);\n")

    table_path = os.path.join(args.out, "cuda_table.frag")
    with open(table_path, "w") as f:
        f.write("// VARIANTS[] entries — splice into VARIANTS[] in main()\n")
        for i, (xk, pu, alt, dens, name, comment) in enumerate(valid):
            td = args.td_base + i
            f.write(f'    VCFG("{name}", {td}, 8),\n')

    py_funcs_path = os.path.join(args.out, "python_funcs.py.frag")
    with open(py_funcs_path, "w") as f:
        f.write('"""sw_<name>() defs — splice into tools/analyze_swizzle.py near sw_gflip_*."""\n')
        f.write("# BEGIN COORD_DESCEND\n\n")
        for xk, pu, alt, dens, name, comment in valid:
            f.write(emit_python(name, xk, pu, alt, dens, comment))
            f.write("\n\n\n")
        f.write("# END COORD_DESCEND\n")

    py_table_path = os.path.join(args.out, "python_table.frag")
    with open(py_table_path, "w") as f:
        f.write('# VARIANTS list entries — splice into the VARIANTS list in analyze_swizzle.py\n')
        for xk, pu, alt, dens, name, comment in valid:
            f.write(f'    ("{name}",  sw_{name}),\n')

    sweep_path = os.path.join(args.out, "sweep_names.txt")
    with open(sweep_path, "w") as f:
        f.write(",".join(name for _, _, _, _, name, _ in valid))
        f.write("\n")

    print(f"wrote: {cuda_path}")
    print(f"wrote: {disp_path}")
    print(f"wrote: {table_path}")
    print(f"wrote: {py_funcs_path}")
    print(f"wrote: {py_table_path}")
    print(f"wrote: {sweep_path}")
    print()
    print("Next steps:")
    print("  1. splice cuda_templates.cuh + cuda_dispatch.frag + cuda_table.frag into fc2_w3x.cu")
    print("  2. splice python_funcs.py.frag + python_table.frag into tools/analyze_swizzle.py")
    print("  3. python3 tools/analyze_swizzle.py --csv data/coord_metrics.csv")
    print("  4. python3 tools/bloom_filter.py data/coord_metrics.csv")
    print(f"  5. SWEEP=$(cat {sweep_path}) REPS=16384 tools/sweep_fc2_w3x_swizzle.sh")


if __name__ == "__main__":
    main()
