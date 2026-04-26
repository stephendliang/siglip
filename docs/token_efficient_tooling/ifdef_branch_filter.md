# Filter #ifdef branches by active DFLAGS

`fc2_w3.cu` is a variant ladder — most branches don't apply to the current
build. Reading all of them wastes tokens on code that isn't compiled.

## Preprocess to the active variant

    nvcc -E -DPACKED_TILES -DLEAN_DISPATCH -DN_STAGES=6 \
        -I. fc2_w3.cu -o /tmp/fc2_active.cu

Output has one concrete variant, no `#ifdef` noise. Still large, but no
branches for kernels you're not measuring.

## Isolate one macro body without preprocessing

    awk '/^#ifdef LEAN_DISPATCH/,/^#endif/' fc2_w3.cu
    awk '/^#if.*USE_STMATRIX/,/^#endif/' fc2_w3.cu

## When it's worth it

- Understanding a specific failing variant on B200.
- Diffing "before vs after" for a `#define` flip.

Skip for: quick symbol lookups (grep is faster), or when the variant labels
themselves are what you need (grep for `#if` lines).
