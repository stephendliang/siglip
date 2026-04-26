# Use git for "what changed" instead of re-reading

To reconstruct recent state, git is denser than a fresh full Read.

## Recent changes to a file

    git log --oneline -20 fc2_w3.cu
    git log --oneline -5 fc2_w3x.cu

## What did a specific commit do

    git show <hash> -- fc2_w3x.cu          # full diff, bounded
    git show --stat <hash>                 # just the files/line counts

## Recent commits touching a topic

    git log --oneline --all --grep='dgswizzle' -20
    git log --oneline -S 'TMA_STORE_WIDE' -- '*.cu'

## When NOT to use git

- Current-state questions ("what does W3 do today?") — read the code; the
  commit message is about the delta, not the present.
- Checking whether a flag exists — grep is faster and authoritative.

## Authoritative vs frozen

`git log`/`git show` is authoritative for history. Memory snapshots of
"current benchmarks" in MEMORY.md decay — prefer git + fresh bench output
for current-state claims.
