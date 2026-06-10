from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
END_WORD = "</w>"


class BPETokenizer:
    """A compact joint BPE tokenizer for English and Chinese text.

    The implementation is intentionally self-contained so the project can run
    offline in a classroom environment. It learns frequent adjacent symbol
    merges, then uses the same subword vocabulary for source and target sides.
    """

    def __init__(self, vocab_size: int = 4000, min_freq: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.merges: list[tuple[str, str]] = []
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.itos: list[str] = list(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def train(self, texts: Iterable[str]) -> None:
        words: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in self._basic_words(text):
                symbols = tuple(list(word) + [END_WORD])
                words[symbols] += 1

        merges: list[tuple[str, str]] = []
        while len(self._symbols(words)) + len(SPECIAL_TOKENS) < self.vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, freq in words.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < self.min_freq:
                break
            words = self._merge_vocab(words, best_pair)
            merges.append(best_pair)

        self.merges = merges
        vocab_tokens = sorted(self._symbols(words), key=lambda x: (-len(x), x))
        self.itos = list(SPECIAL_TOKENS)
        for token in vocab_tokens:
            if token != END_WORD and token not in self.stoi:
                self.itos.append(token)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        pieces: list[str] = []
        for word in self._basic_words(text):
            symbols = list(word) + [END_WORD]
            for pair in self.merges:
                symbols = self._merge_symbols(symbols, pair)
            pieces.extend(tok for tok in symbols if tok != END_WORD)

        ids = [self.stoi.get(tok, self.unk_id) for tok in pieces]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if not ids and add_special:
                ids = [self.eos_id]
            if add_special and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            token = self.itos[int(idx)] if int(idx) < len(self.itos) else UNK
            if token != UNK:
                tokens.append(token)
        text = "".join(tokens)
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" ，", "，").replace(" 。", "。").replace(" ？", "？")
        text = text.replace(" ！", "！").replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
        return text.strip()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "merges": self.merges,
            "itos": self.itos,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["vocab_size"], payload["min_freq"])
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer.itos = payload["itos"]
        tokenizer.stoi = {tok: idx for idx, tok in enumerate(tokenizer.itos)}
        return tokenizer

    @staticmethod
    def _basic_words(text: str) -> list[str]:
        text = text.strip().lower()
        chinese = r"[\u4e00-\u9fff]"
        english_or_num = r"[a-zA-Z0-9]+(?:'[a-z]+)?"
        punct = r"[^\s]"
        return re.findall(f"{chinese}|{english_or_num}|{punct}", text)

    @staticmethod
    def _symbols(words: Counter[tuple[str, ...]]) -> set[str]:
        return {symbol for word in words for symbol in word}

    @staticmethod
    def _merge_symbols(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    @classmethod
    def _merge_vocab(
        cls, words: Counter[tuple[str, ...]], pair: tuple[str, str]
    ) -> Counter[tuple[str, ...]]:
        merged_words: Counter[tuple[str, ...]] = Counter()
        for symbols, freq in words.items():
            merged_words[tuple(cls._merge_symbols(list(symbols), pair))] += freq
        return merged_words
