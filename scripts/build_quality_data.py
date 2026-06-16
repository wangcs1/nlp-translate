from __future__ import annotations

import argparse
import io
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


OPUS_SOURCES = [
    {
        "name": "News-Commentary",
        "url": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-zh.txt.zip",
    },
    {
        "name": "WMT-News",
        "url": "https://object.pouta.csc.fi/OPUS-WMT-News/v2019/moses/en-zh.txt.zip",
    },
]


@dataclass(frozen=True)
class QualityConfig:
    min_src_chars: int
    max_src_chars: int
    min_tgt_chars: int
    max_tgt_chars: int
    min_zh_ratio: float
    max_len_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean EN-ZH TSV corpus from curated OPUS sources.")
    parser.add_argument("--output", default="data/en_zh_quality.tsv")
    parser.add_argument("--max-examples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-src-chars", type=int, default=8)
    parser.add_argument("--max-src-chars", type=int, default=240)
    parser.add_argument("--min-tgt-chars", type=int, default=4)
    parser.add_argument("--max-tgt-chars", type=int, default=180)
    parser.add_argument("--min-zh-ratio", type=float, default=0.35)
    parser.add_argument("--max-len-ratio", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = QualityConfig(
        min_src_chars=args.min_src_chars,
        max_src_chars=args.max_src_chars,
        min_tgt_chars=args.min_tgt_chars,
        max_tgt_chars=args.max_tgt_chars,
        min_zh_ratio=args.min_zh_ratio,
        max_len_ratio=args.max_len_ratio,
    )

    pairs: list[tuple[str, str]] = []
    seen_src: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for source in OPUS_SOURCES:
            print(f"downloading {source['name']} ...")
            zip_path = tmp_root / f"{source['name']}.zip"
            download_file(source["url"], zip_path)
            for src, tgt in read_opus_zip(zip_path):
                src = normalize_english(src)
                tgt = normalize_chinese(tgt)
                if not is_quality_pair(src, tgt, quality):
                    continue
                key = (src.lower(), tgt)
                if key in seen_pair or src.lower() in seen_src:
                    continue
                seen_pair.add(key)
                seen_src.add(src.lower())
                pairs.append((src, tgt))
                if args.max_examples and len(pairs) >= args.max_examples:
                    break
            if args.max_examples and len(pairs) >= args.max_examples:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(f"{src}\t{tgt}" for src, tgt in pairs) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} high-quality sentence pairs to {output}")


def download_file(url: str, output: Path) -> None:
    with urlopen(url) as response:
        output.write_bytes(response.read())


def read_opus_zip(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        src_name = next(name for name in zf.namelist() if name.endswith(".en"))
        tgt_name = next(name for name in zf.namelist() if name.endswith(".zh"))
        with zf.open(src_name) as src_file, zf.open(tgt_name) as tgt_file:
            src_text = io.TextIOWrapper(src_file, encoding="utf-8")
            tgt_text = io.TextIOWrapper(tgt_file, encoding="utf-8")
            reader = zip(src_text, tgt_text)
            for src_line, tgt_line in reader:
                pairs.append((src_line.rstrip("\n"), tgt_line.rstrip("\n")))
    return pairs


def normalize_english(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = " ".join(text.strip().split())
    return normalize_quotes(text)


def normalize_chinese(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", "", text.strip())
    return normalize_quotes(text)


def normalize_quotes(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def is_quality_pair(src: str, tgt: str, quality: QualityConfig) -> bool:
    if "\t" in src or "\t" in tgt:
        return False
    if not (quality.min_src_chars <= len(src) <= quality.max_src_chars):
        return False
    if not (quality.min_tgt_chars <= len(tgt) <= quality.max_tgt_chars):
        return False
    if not re.search(r"[A-Za-z]", src):
        return False
    if has_url_or_markup(src) or has_url_or_markup(tgt):
        return False
    if chinese_ratio(tgt) < quality.min_zh_ratio:
        return False
    if repeated_punctuation(src) or repeated_punctuation(tgt):
        return False
    ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
    return ratio <= quality.max_len_ratio


def has_url_or_markup(text: str) -> bool:
    return bool(re.search(r"https?://|www\.|<[^>]+>|&[a-z]+;", text, flags=re.IGNORECASE))


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh_chars / len(text)


def repeated_punctuation(text: str) -> bool:
    return bool(re.search(r"([!?.,;:，。！？；：])\1{3,}", text))


if __name__ == "__main__":
    main()
