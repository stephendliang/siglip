# Delegate multi-file searches to the Explore agent

Large grep sweeps can return thousands of hits. Routing them through the
Explore subagent means its 200-word summary lands in context, not 5K tokens
of raw ripgrep output.

## When to use Explore

- "Where do we use X across the tree" when X appears in >5 files.
- "How does the epilogue_mbar handshake work" — cross-file trace.
- "What variants does fc1_w3.cu support" — requires walking flag + usage.
- Thoroughness levels: `quick` for direct lookups, `medium` for moderate
  exploration, `very thorough` only when genuinely open-ended.

## When NOT to use Explore

- Known single-file target → `Read` directly.
- One-shot grep you already know the path for → Bash grep.
- Any task that needs to write code → Explore can't Edit.

## Cost tradeoff

Explore spawns a full model call. Worth it when the alternative is pasting
>2K tokens of raw tool output. Not worth it for a single grep.
