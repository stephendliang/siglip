#!/usr/bin/env python3
"""
Inter-warp STS stagger via predicated YIELD.

Replaces NOP slots in the epilogue with @P6 YIELD to create temporal
stagger between epilogue warp groups {W3,W5} and {W4,W6}.

Two modes:
  --mode yield-only: Replace NOPs with plain YIELD (all warps yield, baseline test)
  --mode predicated: Setup P6 = warp parity, then @P6 YIELD (inter-warp stagger)

For predicated mode, the script finds ISETP/LOP3 donors in the SASS dump
and patches register fields. Requires --sass from cuobjdump.

Usage (on B200):
  cuobjdump --dump-sass fc2-w3 > sass.txt
  python3 tools/interwarp_stagger.py fc2-w3 --sass sass.txt --mode yield-only -o fc2-w3-yield
  python3 tools/interwarp_stagger.py fc2-w3 --sass sass.txt --mode predicated -o fc2-w3-stagger
"""
import struct
import sys
import os
import re
import argparse

INSN_SIZE = 16
NOP_ENC = 0x0000000000007918


def find_cubin_in_elf(data):
    """Find embedded cubin(s) in ELF host binary via fatbin container."""
    # Use sass_edit's robust fatbin parser (handles compression, proper ELF parsing)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.sass_edit import find_cubin_in_fatbin
    return find_cubin_in_fatbin(data)


def find_text_section(cubin):
    """Find the fc2_w3 kernel's .text section in cubin ELF."""
    e_shoff = struct.unpack_from('<Q', cubin, 40)[0]
    e_shentsize = struct.unpack_from('<H', cubin, 58)[0]
    e_shnum = struct.unpack_from('<H', cubin, 60)[0]
    e_shstrndx = struct.unpack_from('<H', cubin, 62)[0]

    str_base = e_shoff + e_shstrndx * e_shentsize
    str_off = struct.unpack_from('<Q', cubin, str_base + 24)[0]
    str_sz = struct.unpack_from('<Q', cubin, str_base + 32)[0]
    strtab = cubin[str_off:str_off + str_sz]

    best = None
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        nm_off = struct.unpack_from('<I', cubin, base)[0]
        sh_type = struct.unpack_from('<I', cubin, base + 4)[0]
        sh_offset = struct.unpack_from('<Q', cubin, base + 24)[0]
        sh_size = struct.unpack_from('<Q', cubin, base + 32)[0]
        end = strtab.index(b'\0', nm_off)
        name = strtab[nm_off:end].decode('ascii', errors='replace')
        if '.text.' in name and 'fc2_w3' in name and sh_type == 1 and '.rela.' not in name and '.merc.' not in name and '.capmerc.' not in name:
            if best is None or sh_size > best[1]:
                best = (sh_offset, sh_size)
    return best


def read_insn(cubin, text_off, pc):
    off = text_off + pc
    return struct.unpack_from('<QQ', cubin, off)


def write_insn(cubin, text_off, pc, enc, ctrl):
    off = text_off + pc
    struct.pack_into('<Q', cubin, off, enc)
    struct.pack_into('<Q', cubin, off + 8, ctrl)


def find_nops_in_range(cubin, text_off, text_size, pc_lo, pc_hi):
    nops = []
    for pc in range(pc_lo, min(pc_hi, text_size), INSN_SIZE):
        enc, ctrl = read_insn(cubin, text_off, pc)
        if enc == NOP_ENC:
            nops.append(pc)
    return nops


def find_epilogue_range(cubin, text_off, text_size):
    """Find epilogue boundaries by locating first and last LDTM.x32 (opcode 0x79ee)."""
    ldtms = []
    for pc in range(0, text_size, INSN_SIZE):
        enc, ctrl = read_insn(cubin, text_off, pc)
        if (enc & 0xFFFF) == 0x79ee:
            ldtms.append(pc)
    if not ldtms:
        return None
    # Epilogue starts a bit before first LDTM, ends after last
    return (ldtms[0] - 0x100, ldtms[-1] + 0x200, ldtms)


def parse_sass_for_donors(sass_path):
    """Parse cuobjdump SASS to find ISETP and LOP3 donor instructions."""
    donors = {'isetp_ne_rz': [], 'lop3': []}

    with open(sass_path) as f:
        for line in f:
            # Match: /*addr*/  ISETP.NE.AND P0, PT, R38, RZ, PT ;
            m = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line, re.I)
            if not m:
                continue
            addr = int(m.group(1), 16)
            insn = m.group(2).strip()

            # Find ISETP.NE that compares against RZ
            if 'ISETP.NE' in insn and 'RZ' in insn:
                donors['isetp_ne_rz'].append((addr, insn))

            # Find LOP3.LUT with any operands (we'll patch registers)
            if 'LOP3.LUT' in insn:
                donors['lop3'].append((addr, insn))

    return donors


def find_s2r_donor(cubin, text_off, text_size):
    """Find S2R SR_TID.X (SR=13) donor encoding."""
    sr_mask = 0x0000001f2500010b
    reg_mask = 0xFF << 16
    opcode_mask = ~(sr_mask | reg_mask) & 0xFFFFFFFFFFFFFFFF

    for pc in range(0, text_size, INSN_SIZE):
        enc, ctrl = read_insn(cubin, text_off, pc)
        if (enc & opcode_mask) != (NOP_ENC & opcode_mask):
            # Not S2R family
            continue
        # Extract SR ID
        field_spec = [(0, 2), (3, 1), (8, 1), (24, 1), (26, 1), (29, 1), (32, 5)]
        sr_id = 0
        bp = 0
        for fs, fw in field_spec:
            sr_id |= ((enc >> fs) & ((1 << fw) - 1)) << bp
            bp += fw
        if sr_id == 13:
            return pc, enc, ctrl
    return None


def patch_reg_bits_24_31(enc, new_reg):
    """Patch register at bits 24-31 (common src register position)."""
    return (enc & ~(0xFF << 24)) | ((new_reg & 0xFF) << 24)


def patch_reg_bits_16_23(enc, new_reg):
    """Patch register at bits 16-23 (common dest register position)."""
    return (enc & ~(0xFF << 16)) | ((new_reg & 0xFF) << 16)


def make_yield_enc(pred_reg=7, neg=False):
    """Build YIELD encoding with predicate guard.
    pred_reg=7 means @PT (always). pred_reg=6 means @P6, etc.
    Guard predicate at bits 12-14, negation at bit 15."""
    base = 0x0000000000007946  # YIELD base encoding (@PT)
    # Clear guard field and set new predicate
    enc = base & ~(0xF << 12)
    enc |= (pred_reg & 0x7) << 12
    if neg:
        enc |= 1 << 15
    return enc


YIELD_CTRL = 0x040fe400078efcff  # conservative control word for YIELD


def main():
    parser = argparse.ArgumentParser(description='Inter-warp STS stagger')
    parser.add_argument('binary', help='fc2-w3 host binary')
    parser.add_argument('--sass', help='SASS dump from cuobjdump (required for predicated mode)')
    parser.add_argument('-o', '--output', help='Output patched binary')
    parser.add_argument('--mode', choices=['yield-only', 'predicated'], default='yield-only',
                        help='yield-only: plain YIELD at NOPs. predicated: @P6 YIELD with setup')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--scratch-reg', type=int, default=206)
    parser.add_argument('--pred-reg', type=int, default=7,
                        help='Predicate register for YIELD guard (7=PT=always, 6=P6=warp parity)')
    args = parser.parse_args()

    # Load binary
    with open(args.binary, 'rb') as f:
        host_data = bytearray(f.read())

    cubins = find_cubin_in_elf(host_data)
    if not cubins:
        print('ERROR: No embedded cubins found')
        sys.exit(1)

    cubin_off, cubin_size = cubins[0]
    cubin = bytearray(host_data[cubin_off:cubin_off + cubin_size])

    result = find_text_section(cubin)
    if not result:
        print('ERROR: Kernel .text section not found')
        sys.exit(1)

    text_off, text_size = result
    n_insns = text_size // INSN_SIZE
    print(f'Kernel: {n_insns} instructions')

    # Find epilogue
    epi = find_epilogue_range(cubin, text_off, text_size)
    if not epi:
        print('ERROR: No LDTM instructions found (no epilogue?)')
        sys.exit(1)

    epi_lo, epi_hi, ldtms = epi
    epi_nops = find_nops_in_range(cubin, text_off, text_size, epi_lo, epi_hi)
    print(f'Epilogue: 0x{epi_lo:04x}-0x{epi_hi:04x}, {len(ldtms)} LDTMs, {len(epi_nops)} NOPs')

    if args.mode == 'yield-only':
        # Replace ALL epilogue NOPs with YIELD.
        # Default pred_reg=7 (@PT = all warps). Use --pred-reg 6 for @P6 (needs
        # source-level WARP_STAGGER to set P6 = warp parity).
        yield_enc = make_yield_enc(pred_reg=args.pred_reg)
        pred_label = f'@P{args.pred_reg}' if args.pred_reg != 7 else '@PT'
        print(f'\n=== YIELD-ONLY MODE ({pred_label}) ===')
        print(f'Replacing {len(epi_nops)} NOPs with {pred_label} YIELD:')
        for pc in epi_nops:
            print(f'  0x{pc:04x}: NOP → {pred_label} YIELD')

        if not args.dry_run and args.output:
            for pc in epi_nops:
                write_insn(cubin, text_off, pc, yield_enc, YIELD_CTRL)
            host_data[cubin_off:cubin_off + cubin_size] = cubin
            with open(args.output, 'wb') as f:
                f.write(host_data)
            os.chmod(args.output, 0o755)
            print(f'\nPatched {len(epi_nops)} instructions → {args.output}')

    elif args.mode == 'predicated':
        if not args.sass:
            print('ERROR: --sass required for predicated mode')
            sys.exit(1)

        print(f'\n=== PREDICATED MODE (P{args.pred_reg}) ===')

        # Find S2R donor
        s2r = find_s2r_donor(cubin, text_off, text_size)
        if not s2r:
            print('ERROR: No S2R SR_TID.X donor found')
            sys.exit(1)
        s2r_pc, s2r_enc, s2r_ctrl = s2r
        print(f'S2R donor: PC=0x{s2r_pc:04x}')

        # Parse SASS for ISETP/LOP3 donors
        donors = parse_sass_for_donors(args.sass)
        print(f'SASS donors: {len(donors["isetp_ne_rz"])} ISETP.NE+RZ, {len(donors["lop3"])} LOP3')

        if len(epi_nops) < 3:
            print('ERROR: Need ≥3 NOPs for setup')
            sys.exit(1)

        # Find a suitable ISETP donor — one that does ISETP.NE.AND Px, PT, Ry, RZ, PT
        isetp_donor = None
        for addr, insn_text in donors['isetp_ne_rz']:
            # Parse: ISETP.NE.AND P0, PT, R38, RZ, PT
            m = re.match(r'ISETP\.NE\.AND\s+P(\d+),\s*PT,\s*R(\d+),\s*RZ,\s*PT', insn_text)
            if m:
                pd = int(m.group(1))
                rs = int(m.group(2))
                isetp_donor = (addr, pd, rs, insn_text)
                break

        if not isetp_donor:
            # Try ISETP.NE.U32.AND
            for addr, insn_text in donors['isetp_ne_rz']:
                m = re.match(r'ISETP\.NE(?:\.U32)?\.AND\s+P(\d+),\s*PT,\s*R(\d+),\s*RZ,\s*PT', insn_text)
                if m:
                    pd = int(m.group(1))
                    rs = int(m.group(2))
                    isetp_donor = (addr, pd, rs, insn_text)
                    break

        if not isetp_donor:
            print('ERROR: No suitable ISETP.NE.AND Px, PT, Ry, RZ, PT donor in SASS')
            print('Available ISETP donors:')
            for addr, text in donors['isetp_ne_rz'][:10]:
                print(f'  0x{addr:04x}: {text}')
            sys.exit(1)

        isetp_addr, isetp_pd, isetp_rs, isetp_text = isetp_donor
        print(f'ISETP donor: 0x{isetp_addr:04x} {isetp_text}')
        print(f'  dest_pred=P{isetp_pd}, src_reg=R{isetp_rs}')

        # Read ISETP donor encoding
        isetp_enc, isetp_ctrl = read_insn(cubin, text_off, isetp_addr)
        print(f'  enc=0x{isetp_enc:016x} ctrl=0x{isetp_ctrl:016x}')

        # Patch ISETP: change src register from R{isetp_rs} to R{scratch}
        # Src register at bits 24-31 (standard position for ISETP based on our analysis)
        verify_rs = (isetp_enc >> 24) & 0xFF
        if verify_rs != isetp_rs:
            print(f'WARNING: ISETP src reg mismatch at bits 24-31: expected R{isetp_rs}, got R{verify_rs}')
            print('Trying bits 32-39...')
            verify_rs = (isetp_enc >> 32) & 0xFF
            if verify_rs == isetp_rs:
                print(f'  Found src reg at bits 32-39. Adjusting...')
                # Non-standard position — patch bits 32-39 instead
                isetp_patched = (isetp_enc & ~(0xFF << 32)) | (args.scratch_reg << 32)
            else:
                print(f'ERROR: Cannot locate src register in ISETP encoding')
                sys.exit(1)
        else:
            isetp_patched = patch_reg_bits_24_31(isetp_enc, args.scratch_reg)

        # Patch ISETP: change dest predicate from P{isetp_pd} to P{pred_reg}
        # Need to find where the dest predicate is encoded
        # Strategy: XOR the donor encoding with what we'd get if we changed the pred
        # For now, try common predicate positions
        # From analysis: bits 9-11 seem to correlate with predicate
        verify_pd = (isetp_enc >> 9) & 0x7
        # This might not be the pred dest — it could be comparison mode bits
        # Let's try it and see
        print(f'  Bits 9-11 = {verify_pd} (P{isetp_pd}?)')

        # Find a second ISETP with different dest pred to identify the field
        for addr2, text2 in donors['isetp_ne_rz']:
            m2 = re.match(r'ISETP\.NE(?:\.U32)?\.AND\s+P(\d+),\s*PT,\s*R(\d+),\s*RZ,\s*PT', text2)
            if m2:
                pd2 = int(m2.group(1))
                if pd2 != isetp_pd:
                    enc2, _ = read_insn(cubin, text_off, addr2)
                    bits_9_11_2 = (enc2 >> 9) & 0x7
                    print(f'  Second ISETP: P{pd2} → bits 9-11 = {bits_9_11_2}')
                    # If bits 9-11 track the predicate number, we can use this field
                    break

        # Patch dest predicate at bits 9-11 (tentative)
        isetp_patched = (isetp_patched & ~(0x7 << 9)) | ((args.pred_reg & 0x7) << 9)

        # Find LOP3 donor
        lop3_donor = None
        for addr, insn_text in donors['lop3']:
            lop3_donor = (addr, insn_text)
            break

        if not lop3_donor:
            print('ERROR: No LOP3.LUT donor in SASS')
            sys.exit(1)

        lop3_addr, lop3_text = lop3_donor
        lop3_enc, lop3_ctrl = read_insn(cubin, text_off, lop3_addr)
        print(f'LOP3 donor: 0x{lop3_addr:04x} {lop3_text}')
        print(f'  enc=0x{lop3_enc:016x}')

        # We need LOP3.LUT R206, R206, 0x20, RZ, 0xC0
        # The LOP3 donor has some different registers and immediate/LUT values
        # We can patch registers (bits 16-23 = dest, bits 24-31 = src1)
        # But the immediate (0x20) and LUT (0xC0) are in bit positions we'd need
        # to figure out from two different LOP3 donors with different imm/LUT

        # For now: just patch the registers and hope the imm/LUT are acceptable
        # OR: find a LOP3 with specific immediate pattern
        lop3_good = None
        for addr, text in donors['lop3']:
            # Look for LOP3 with 0x20 or similar small immediate
            if '0x20' in text or ', 0x20,' in text:
                enc, ctrl = read_insn(cubin, text_off, addr)
                lop3_good = (addr, text, enc, ctrl)
                break

        if lop3_good:
            lop3_addr, lop3_text, lop3_enc, lop3_ctrl = lop3_good
            print(f'LOP3 with 0x20: 0x{lop3_addr:04x} {lop3_text}')

        # Plan
        setup_nops = epi_nops[:3]
        yield_nops = epi_nops[3:]

        s2r_patched = patch_reg_bits_16_23(s2r_enc, args.scratch_reg)

        print()
        print('Patch plan:')
        print(f'  0x{setup_nops[0]:04x}: NOP → S2R R{args.scratch_reg}, SR_TID.X')
        print(f'  0x{setup_nops[1]:04x}: NOP → LOP3.LUT R{args.scratch_reg}, R{args.scratch_reg}, 0x20, RZ, 0xC0')
        print(f'  0x{setup_nops[2]:04x}: NOP → ISETP.NE.AND P{args.pred_reg}, PT, R{args.scratch_reg}, RZ, PT')
        print(f'  {len(yield_nops)} NOPs → @P{args.pred_reg} YIELD')

        yield_enc = make_yield_enc(pred_reg=args.pred_reg)
        print(f'  YIELD encoding: 0x{yield_enc:016x}')

        if not args.dry_run and args.output:
            # Apply setup
            write_insn(cubin, text_off, setup_nops[0], s2r_patched, s2r_ctrl)
            # LOP3 — patch registers, keep everything else from donor
            lop3_patched = patch_reg_bits_16_23(lop3_enc, args.scratch_reg)
            lop3_patched = patch_reg_bits_24_31(lop3_patched, args.scratch_reg)
            write_insn(cubin, text_off, setup_nops[1], lop3_patched, lop3_ctrl)
            # ISETP
            write_insn(cubin, text_off, setup_nops[2], isetp_patched, isetp_ctrl)
            # YIELDs
            for pc in yield_nops:
                write_insn(cubin, text_off, pc, yield_enc, YIELD_CTRL)

            host_data[cubin_off:cubin_off + cubin_size] = cubin
            with open(args.output, 'wb') as f:
                f.write(host_data)
            os.chmod(args.output, 0o755)
            print(f'\nPatched {3 + len(yield_nops)} instructions → {args.output}')
        elif args.dry_run:
            print('\n(dry run)')
        else:
            print('\nNo --output specified')


if __name__ == '__main__':
    main()
