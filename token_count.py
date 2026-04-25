#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import tiktoken

ENCODINGS = [
    ("o200k_base", "GPT-4o / GPT-4.1 / o-series"),
    ("cl100k_base", "GPT-4 / GPT-3.5-turbo / text-embedding-3"),
]


def count_tiktoken(text: str, encoding_name: str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text, disallowed_special=()))


def count_chars_div4(text: str) -> int:
    return (len(text) + 3) // 4


def count_words_scaled(text: str, factor: float = 1.3) -> int:
    return int(len(text.split()) * factor)


def count_whitespace_split(text: str) -> int:
    return len(text.split())


def human(n: int) -> str:
    for unit in ("", "K", "M"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if isinstance(n, float) else f"{n}{unit}"
        n /= 1000
    return f"{n:.1f}G"


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate token count for a file across multiple tokenizers.")
    ap.add_argument("path", help="file to analyze (use '-' for stdin)")
    ap.add_argument("--encoding", default="utf-8", help="text encoding of the input file (default: utf-8)")
    ap.add_argument("--only", help="comma-separated list of tiktoken encodings to limit to")
    args = ap.parse_args()

    if args.path == "-":
        text = sys.stdin.read()
        src = "<stdin>"
        nbytes = len(text.encode(args.encoding, errors="replace"))
    else:
        p = Path(args.path)
        if not p.is_file():
            print(f"error: not a file: {p}", file=sys.stderr)
            return 2
        raw = p.read_bytes()
        nbytes = len(raw)
        text = raw.decode(args.encoding, errors="replace")
        src = str(p)

    nchars = len(text)
    nlines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

    print(f"file:  {src}")
    print(f"bytes: {nbytes}  chars: {nchars}  lines: {nlines}")
    print()

    wanted = None
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"{'method':<28} {'tokens':>10}  {'bytes/tok':>10}  {'chars/tok':>10}  note")
    print("-" * 90)

    for name, note in ENCODINGS:
        if wanted is not None and name not in wanted:
            continue
        try:
            n = count_tiktoken(text, name)
        except Exception as e:
            print(f"{name:<28} {'ERR':>10}  {'':>10}  {'':>10}  {e}")
            continue
        bpt = nbytes / n if n else 0.0
        cpt = nchars / n if n else 0.0
        print(f"{name:<28} {n:>10d}  {bpt:>10.2f}  {cpt:>10.2f}  {note}")

    if wanted is None:
        print()
        heur = [
            ("chars / 4 (rule of thumb)", count_chars_div4(text), "classic OpenAI heuristic"),
            ("words * 1.3", count_words_scaled(text), "whitespace words scaled"),
            ("whitespace words", count_whitespace_split(text), "raw word count"),
        ]
        for label, n, note in heur:
            bpt = nbytes / n if n else 0.0
            cpt = nchars / n if n else 0.0
            print(f"{label:<28} {n:>10d}  {bpt:>10.2f}  {cpt:>10.2f}  {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
