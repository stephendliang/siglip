#!/bin/bash
# Benchmark sweep over various imgs_per_sm (controls M)
# M = imgs_per_sm * 148 SMs * 196 seq_len

echo "═══════════════════════════════════════════════════════════"
echo "  SigLIP2 Patch Embed GEMM — M-size sweep on B200"
echo "  N=768, K=768, TM=128, TN=128, TK=32, 3-stage pipeline"
echo "═══════════════════════════════════════════════════════════"
echo ""
printf "%-12s %-12s %-12s %-12s\n" "imgs/SM" "M" "ms" "TFLOPS"
echo "---------------------------------------------------"

for ips in 1 2 4 8 16 32; do
    M=$((ips * 148 * 196))
    python3 gen.py --no-coop --imgs-per-sm $ips --warmup 2 --iters 10 -o megakernel.cu 2>/dev/null
    make siglip_vision 2>/dev/null
    result=$(./siglip_vision 2>&1)
    ms=$(echo "$result" | grep "Custom kernel:" | awk '{print $3}')
    tflops=$(echo "$result" | grep "Custom kernel:" | awk '{print $5}')
    printf "%-12d %-12d %-12s %-12s\n" "$ips" "$M" "$ms" "$tflops"
done

echo ""
echo "Done."
