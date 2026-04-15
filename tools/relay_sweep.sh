#!/bin/bash
set -e
for kb in 16 32 48 64; do
    sz=$((kb * 1024))
    make -sB relay-mbar DFLAGS="-DXFER=$sz" 2>/dev/null
    echo "════════════════════════ XFER = ${kb} KB ════════════════════════"
    ./relay-mbar
    echo
done
