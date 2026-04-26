# Cache a kernel structural map

One-time map of a 4000-line kernel → 100-line index. Read the map first,
then narrow-Read the kernel at the exact offset.

## Generate

    {
      echo "# fc2_w3.cu structural map"
      echo
      grep -nE '^(void|__device__|__global__|static|__forceinline__)' fc2_w3.cu
      echo
      echo "## Variant gates"
      grep -nE '^#ifdef|^#ifndef|^#if ' fc2_w3.cu
      echo
      echo "## Warp role markers"
      grep -nE '// W[0-9]|warp_id ==' fc2_w3.cu
    } > docs/token_efficient_tooling/kernel_map_fc2_w3.md

## Refresh cadence

After any structural refactor (new warp, reordered sections, new variant
flag). Not after variant tuning — the map is position-stable.

## What it replaces

Reading `fc2_w3.cu` start-to-end (~57K tokens) just to find where W3's
epilogue loop lives. Map + `Read offset=X limit=60` = ~500 tokens.
