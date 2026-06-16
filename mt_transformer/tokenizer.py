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


class BPETokenizer:
    """A compact shared tokenizer for English and Chinese text.

    The name is kept for compatibility with the rest of the project, but the
    implementation is now a fast shared vocabulary built from the training
    corpus. English words, numbers, Chinese characters, and punctuation are
    all tokenized with the same vocabulary.
    """

    def __init__(self, vocab_size: int = 4000, min_freq: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_freq = min_freq
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
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._basic_tokens(text))

        vocab_tokens = [
            token
            for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if freq >= self.min_freq and token not in SPECIAL_TOKENS
        ]
        limit = max(0, self.vocab_size - len(SPECIAL_TOKENS))
        self.itos = list(SPECIAL_TOKENS) + vocab_tokens[:limit]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(tok, self.unk_id) for tok in self._basic_tokens(text)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if add_special and ids and ids[-1] != self.eos_id:
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
        text = self._join_tokens(tokens)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "itos": self.itos,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["vocab_size"], payload["min_freq"])
        tokenizer.itos = payload["itos"]
        tokenizer.stoi = {tok: idx for idx, tok in enumerate(tokenizer.itos)}
        return tokenizer

    @staticmethod
    def _basic_tokens(text: str) -> list[str]:
        text = text.strip().lower()
        chinese = r"[\u4e00-\u9fff]"
        english_or_num = r"[a-zA-Z0-9]+(?:'[a-z]+)?"
        punct = r"[^\s]"
        return re.findall(f"{chinese}|{english_or_num}|{punct}", text)

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        pieces: list[str] = []
        for token in tokens:
            if not pieces:
                pieces.append(token)
                continue
            prev = pieces[-1]
            if re.fullmatch(r"[\u4e00-\u9fff]", token):
                pieces.append(token)
            elif re.fullmatch(r"[A-Za-z0-9]+(?:'[a-z]+)?", token):
                if re.fullmatch(r"[A-Za-z0-9]+(?:'[a-z]+)?", prev):
                    pieces.append(" " + token)
                else:
                    pieces.append(token)
            else:
                if token in {".", ",", "?", "!", ":", ";", ")", "]", "}"}:
                    pieces.append(token)
                elif token in {"(", "[", "{"}:
                    pieces.append(" " + token)
                else:
                    pieces.append(token)
        return "".join(pieces)
