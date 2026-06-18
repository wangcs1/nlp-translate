from __future__ import annotations

import argparse
import io
import random
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode


OPUS_SOURCES = [
    {
        "name": "News-Commentary",
        "url": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-zh.txt.zip",
        "quota": 40000,
    },
    {
        "name": "WMT-News",
        "url": "https://object.pouta.csc.fi/OPUS-WMT-News/v2019/moses/en-zh.txt.zip",
        "quota": 20000,
    },
    {
        "name": "WikiMatrix",
        "url": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-zh.txt.zip",
        "quota": 50000,
    },
    {
        "name": "MultiUN",
        "url": "https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-zh.txt.zip",
        "quota": 30000,
    },
    {
        "name": "QED",
        "url": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-zh.txt.zip",
        "quota": 5000,
    },
]


TECH_SEED_PAIRS = [
    ("Transformer models learn from parallel data.", "Transformer模型从平行数据中学习。"),
    ("Machine translation converts text from one language into another.", "机器翻译将文本从一种语言转换为另一种语言。"),
    ("Parallel corpora contain aligned source and target sentences.", "平行语料包含对齐的源语言句子和目标语言句子。"),
    ("The tokenizer maps text into a sequence of subword tokens.", "分词器将文本映射为子词标记序列。"),
    ("The encoder reads the source sentence.", "编码器读取源句子。"),
    ("The decoder generates the target sentence step by step.", "解码器逐步生成目标句子。"),
    ("Attention helps the model focus on important words.", "注意力机制帮助模型关注重要词语。"),
    ("Beam search keeps several candidate translations.", "束搜索会保留多个候选译文。"),
    ("The model is trained on English and Chinese sentence pairs.", "模型使用英文和中文句对进行训练。"),
    ("Validation loss is used to select the best checkpoint.", "验证损失用于选择最佳检查点。"),
    ("A larger and cleaner dataset usually improves translation quality.", "更大且更干净的数据集通常会提升翻译质量。"),
    ("Gradient clipping keeps Transformer training stable.", "梯度裁剪可以保持Transformer训练稳定。"),
    ("Label smoothing makes the model less overconfident.", "标签平滑可以降低模型的过度自信。"),
    ("The vocabulary should cover common words in the target domain.", "词表应该覆盖目标领域中的常用词。"),
    ("Out of vocabulary words can hurt translation quality.", "词表外单词会损害翻译质量。"),
    ("Domain mismatch can make a trained model produce strange translations.", "领域不匹配会导致训练好的模型产生奇怪译文。"),
    ("News data is useful for formal written Chinese.", "新闻数据适合训练正式书面中文。"),
    ("Technical sentences require technical parallel data.", "技术句子需要技术领域的平行数据。"),
    ("The checkpoint stores the learned model weights.", "检查点保存学到的模型权重。"),
    ("The training script saves the tokenizer and the best model.", "训练脚本会保存分词器和最佳模型。"),
    ("A third way for confronting Russia requires diplomacy and deterrence.", "应对俄罗斯的第三条道路需要外交和威慑。"),
    ("The committee approved the reform package after a long debate.", "委员会在长时间辩论后批准了改革方案。"),
    ("The global economy is facing serious challenges.", "全球经济正面临严峻挑战。"),
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
    parser = argparse.ArgumentParser(description="Build a broad clean EN-ZH TSV corpus from OPUS sources.")
    parser.add_argument("--output", default="data/en_zh_quality.tsv")
    parser.add_argument("--cache-dir", default="data/raw_opus")
    parser.add_argument("--max-examples", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-src-chars", type=int, default=4)
    parser.add_argument("--max-src-chars", type=int, default=260)
    parser.add_argument("--min-tgt-chars", type=int, default=2)
    parser.add_argument("--max-tgt-chars", type=int, default=220)
    parser.add_argument("--min-zh-ratio", type=float, default=0.25)
    parser.add_argument("--max-len-ratio", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    quality = QualityConfig(
        min_src_chars=args.min_src_chars,
        max_src_chars=args.max_src_chars,
        min_tgt_chars=args.min_tgt_chars,
        max_tgt_chars=args.max_tgt_chars,
        min_zh_ratio=args.min_zh_ratio,
        max_len_ratio=args.max_len_ratio,
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, str]] = []
    seen_src: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()

    for src, tgt in TECH_SEED_PAIRS:
        add_pair(src, tgt, pairs, seen_src, seen_pair)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        total_quota = sum(source["quota"] for source in OPUS_SOURCES)
        scale = min(1.0, args.max_examples / total_quota) if args.max_examples else 1.0
        for source in OPUS_SOURCES:
            print(f"loading {source['name']} ...")
            try:
                zip_path = cached_download(source, cache_dir, tmp_root)
            except Exception as exc:
                print(f"  skipped {source['name']}: {exc}")
                continue
            source_pairs: list[tuple[str, str]] = []
            for src, tgt in read_opus_zip(zip_path):
                src = normalize_english(src)
                tgt = normalize_chinese(tgt)
                if is_quality_pair(src, tgt, quality):
                    source_pairs.append((src, tgt))
            rng.shuffle(source_pairs)

            added = 0
            source_quota = max(1, int(source["quota"] * scale))
            for src, tgt in source_pairs:
                before = len(pairs)
                add_pair(src, tgt, pairs, seen_src, seen_pair)
                if len(pairs) > before:
                    added += 1
                if added >= source_quota:
                    break
            print(f"  kept {added} pairs from {source['name']}")

    rng.shuffle(pairs)
    if args.max_examples and len(pairs) > args.max_examples:
        pairs = pairs[: args.max_examples]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(f"{src}\t{tgt}" for src, tgt in pairs) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} sentence pairs to {output}")


def cached_download(source: dict, cache_dir: Path, tmp_root: Path) -> Path:
    cached = cache_dir / f"{source['name']}.zip"
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    target = tmp_root / f"{source['name']}.zip"
    url = source["url"] or find_opus_url(source["name"])
    download_file(url, target)
    cached.write_bytes(target.read_bytes())
    return cached


def find_opus_url(corpus: str) -> str:
    query = urlencode({"corpus": corpus, "source": "en", "target": "zh", "preprocessing": "moses"})
    with urlopen(f"https://opus.nlpl.eu/opusapi?{query}") as response:
        import json

        payload = json.loads(response.read().decode("utf-8"))
    corpora = payload.get("corpora", [])
    if not corpora:
        raise ValueError(f"No OPUS moses en-zh corpus found for {corpus}.")
    latest = [item for item in corpora if str(item.get("latest")).lower() == "true"]
    selected = latest[-1] if latest else corpora[-1]
    return selected["url"]


def add_pair(
    src: str,
    tgt: str,
    pairs: list[tuple[str, str]],
    seen_src: set[str],
    seen_pair: set[tuple[str, str]],
) -> None:
    src = normalize_english(src)
    tgt = normalize_chinese(tgt)
    key = (src.lower(), tgt)
    if key in seen_pair or src.lower() in seen_src:
        return
    seen_pair.add(key)
    seen_src.add(src.lower())
    pairs.append((src, tgt))


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
            for src_line, tgt_line in zip(src_text, tgt_text):
                pairs.append((src_line.rstrip("\n"), tgt_line.rstrip("\n")))
    return pairs


def normalize_english(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = " ".join(text.strip().split())
    return normalize_quotes(text)


def normalize_chinese(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = " ".join(text.strip().split())
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
    if not re.search(r"[\u4e00-\u9fff]", tgt):
        return False
    if has_url_or_markup(src) or has_url_or_markup(tgt):
        return False
    if chinese_ratio(tgt) < quality.min_zh_ratio:
        return False
    if latin_ratio(tgt) > 0.45:
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


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    return latin_chars / len(text)


def repeated_punctuation(text: str) -> bool:
    return bool(re.search(r"([!?.,;:，。！？；：])\1{3,}", text))


if __name__ == "__main__":
    main()
