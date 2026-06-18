from __future__ import annotations

import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterable


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


class BPETokenizer:
    """SentencePiece unigram tokenizer with the old class name kept for compatibility."""

    def __init__(self, vocab_size: int = 16000, min_freq: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.model_proto: bytes | None = None
        self.sp = None

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def itos(self) -> list[str]:
        processor = self._processor()
        return [processor.id_to_piece(i) for i in range(processor.get_piece_size())]

    def train(self, texts: Iterable[str]) -> None:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise SystemExit("Please install sentencepiece first: pip install -r requirements.txt") from exc
        if hasattr(spm, "set_min_log_level"):
            spm.set_min_log_level(2)
        elif hasattr(spm, "SetMinLogLevel"):
            spm.SetMinLogLevel(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            corpus_path = tmp_root / "spm_corpus.txt"
            model_prefix = tmp_root / "spm"
            corpus_path.write_text("\n".join(str(text).strip() for text in texts if str(text).strip()), encoding="utf-8")
            vocab_size = max(self.vocab_size, 4000)
            with (
                open(os.devnull, "w", encoding="utf-8") as devnull,
                redirect_stdout(devnull),
                redirect_stderr(devnull),
            ):
                spm.SentencePieceTrainer.train(
                    input=str(corpus_path),
                    model_prefix=str(model_prefix),
                    vocab_size=vocab_size,
                    model_type="unigram",
                    character_coverage=0.9995,
                    pad_id=self.pad_id,
                    bos_id=self.bos_id,
                    eos_id=self.eos_id,
                    unk_id=self.unk_id,
                    pad_piece=PAD,
                    bos_piece=BOS,
                    eos_piece=EOS,
                    unk_piece=UNK,
                    split_by_unicode_script=True,
                    byte_fallback=True,
                    train_extremely_large_corpus=True,
                    hard_vocab_limit=False,
                    input_sentence_size=1000000,
                    shuffle_input_sentence=True,
                )
            self.model_proto = (model_prefix.with_suffix(".model")).read_bytes()
        self.sp = self._load_processor(self.model_proto)

    def encode(self, text: str, add_special: bool = True, max_len: int | None = None) -> list[int]:
        ids = list(self._processor().encode(str(text), out_type=int))
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
            if add_special and ids and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        clean_ids = [int(idx) for idx in ids if int(idx) not in (self.pad_id, self.bos_id, self.eos_id)]
        return self._processor().decode(clean_ids).strip()

    def save(self, path: str | Path) -> None:
        if self.model_proto is None:
            raise ValueError("Tokenizer has not been trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "sentencepiece_unigram",
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "model_hex": self.model_proto.hex(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["vocab_size"], payload.get("min_freq", 1))
        tokenizer.model_proto = bytes.fromhex(payload["model_hex"])
        tokenizer.sp = tokenizer._load_processor(tokenizer.model_proto)
        return tokenizer

    def _processor(self):
        if self.sp is None:
            if self.model_proto is None:
                raise ValueError("Tokenizer has not been trained.")
            self.sp = self._load_processor(self.model_proto)
        return self.sp

    @staticmethod
    def _load_processor(model_proto: bytes):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise SystemExit("Please install sentencepiece first: pip install -r requirements.txt") from exc
        processor = spm.SentencePieceProcessor()
        processor.LoadFromSerializedProto(model_proto)
        return processor
