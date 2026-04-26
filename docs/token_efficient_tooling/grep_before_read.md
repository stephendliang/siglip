# Grep before Read on kernel files

`fc2_w3.cu` is ~57K tokens. Never bulk-read it. Always locate first, then
narrow-read with `offset`/`limit`.

Rule: if you don't have a line number yet, you're not ready to Read.

## Useful one-liners

Symbol/macro hits:
    grep -nE 'TMA_STORE_WIDE|LEAN_DISPATCH|PACKED_TILES' fc2_w3.cu

Function index:
    grep -nE '^__device__|^__global__|^static|^void ' fc2_w3.cu

Variant map (which `#ifdef`s exist):
    grep -nE '^#(if|ifdef|ifndef|elif|else|endif)' fc2_w3.cu

Cross-file where-used:
    grep -rn 'LDTM\.16dp256' --include='*.cu' --include='*.cuh'

## When to use Explore instead

If the answer requires walking 5+ files or multiple naming conventions,
spawn Explore (thoroughness=quick). Its summary is ~200 words vs ~5K tokens
of raw ripgrep hits.
