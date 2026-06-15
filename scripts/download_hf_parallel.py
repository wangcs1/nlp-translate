from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face translation dataset and export TSV.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name, e.g. Helsinki-NLP/opus-100.")
    parser.add_argument("--config", help="Dataset config name, e.g. en-zh.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--src-key", default="src", help="Fallback source column name if no translation field exists.")
    parser.add_argument("--tgt-key", default="tgt", help="Fallback target column name if no translation field exists.")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--min-src-chars", type=int, default=2)
    parser.add_argument("--min-tgt-chars", type=int, default=1)
    parser.add_argument("--max-src-chars", type=int, default=280)
    parser.add_argument("--max-tgt-chars", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Please install datasets first: pip install -r requirements.txt") from exc

    dataset = load_dataset(args.dataset, args.config, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in dataset:
        src, tgt = extract_pair(row, args.source_lang, args.target_lang, args.src_key, args.tgt_key)
        if src is None or tgt is None:
            continue
        src = normalize(src)
        tgt = normalize(tgt)
        if not (args.min_src_chars <= len(src) <= args.max_src_chars):
            continue
        if not (args.min_tgt_chars <= len(tgt) <= args.max_tgt_chars):
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(f"{src}\t{tgt}" for src, tgt in pairs) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} sentence pairs to {output}")


def extract_pair(
    row: dict,
    source_lang: str,
    target_lang: str,
    src_key: str,
    tgt_key: str,
) -> tuple[str | None, str | None]:
    translation = row.get("translation")
    if isinstance(translation, dict):
        return translation.get(source_lang), translation.get(target_lang)
    if src_key in row and tgt_key in row:
        return row[src_key], row[tgt_key]
    return None, None


def normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


if __name__ == "__main__":
    main()
