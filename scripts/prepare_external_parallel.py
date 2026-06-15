from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize external EN-ZH parallel corpora into TSV.")
    parser.add_argument("--src-file", help="English text file, one sentence per line.")
    parser.add_argument("--tgt-file", help="Chinese text file, one sentence per line.")
    parser.add_argument("--input-tsv", help="Existing TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV path: English<TAB>Chinese.")
    parser.add_argument("--src-col", type=int, default=0, help="Source column for --input-tsv, zero-based.")
    parser.add_argument("--tgt-col", type=int, default=1, help="Target column for --input-tsv, zero-based.")
    parser.add_argument("--max-src-chars", type=int, default=280)
    parser.add_argument("--max-tgt-chars", type=int, default=220)
    parser.add_argument("--min-src-chars", type=int, default=2)
    parser.add_argument("--min-tgt-chars", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Randomly sample this many sentence pairs after filtering.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args)
    pairs = filter_pairs(pairs, args)

    if args.limit and len(pairs) > args.limit:
        rng = random.Random(args.seed)
        pairs = rng.sample(pairs, args.limit)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(f"{src}\t{tgt}" for src, tgt in pairs) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} sentence pairs to {output}")


def load_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.input_tsv:
        return load_tsv(Path(args.input_tsv), args.src_col, args.tgt_col)

    if not args.src_file or not args.tgt_file:
        raise SystemExit("Use either --input-tsv or both --src-file and --tgt-file.")

    src_lines = Path(args.src_file).read_text(encoding="utf-8").splitlines()
    tgt_lines = Path(args.tgt_file).read_text(encoding="utf-8").splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Line count mismatch: {len(src_lines)} source vs {len(tgt_lines)} target.")
    return list(zip(src_lines, tgt_lines))


def load_tsv(path: Path, src_col: int, tgt_col: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if max(src_col, tgt_col) >= len(parts):
            raise ValueError(f"Line {line_no} has only {len(parts)} columns.")
        pairs.append((parts[src_col], parts[tgt_col]))
    return pairs


def filter_pairs(pairs: list[tuple[str, str]], args: argparse.Namespace) -> list[tuple[str, str]]:
    clean: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src, tgt in pairs:
        src = normalize(src)
        tgt = normalize(tgt)
        if not (args.min_src_chars <= len(src) <= args.max_src_chars):
            continue
        if not (args.min_tgt_chars <= len(tgt) <= args.max_tgt_chars):
            continue
        if "\t" in src or "\t" in tgt:
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        clean.append(key)
    return clean


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


if __name__ == "__main__":
    main()
