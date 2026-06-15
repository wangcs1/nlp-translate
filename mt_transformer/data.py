from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tokenizer import BPETokenizer


@dataclass
class PairExample:
    src: str
    tgt: str


def read_parallel_tsv(path: str | Path) -> list[PairExample]:
    examples: list[PairExample] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Line {line_no} must be '<english>\\t<chinese>'.")
        examples.append(PairExample(parts[0].strip(), parts[1].strip()))
    return examples


def split_examples(
    examples: list[PairExample], valid_ratio: float = 0.15, seed: int = 42
) -> tuple[list[PairExample], list[PairExample]]:
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    valid_size = max(1, int(len(examples) * valid_ratio)) if len(examples) > 3 else 1
    return examples[valid_size:], examples[:valid_size]


class TranslationDataset(Dataset):
    def __init__(self, examples: list[PairExample], tokenizer: BPETokenizer, max_len: int = 96) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.examples[idx]
        src = torch.tensor(self.tokenizer.encode(item.src, max_len=self.max_len), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(item.tgt, max_len=self.max_len), dtype=torch.long)
        return src, tgt


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        src, tgt = zip(*batch)
        src_batch = pad_sequence(src, batch_first=True, padding_value=pad_id)
        tgt_batch = pad_sequence(tgt, batch_first=True, padding_value=pad_id)
        return src_batch, tgt_batch

    return collate


def build_loader(
    examples: list[PairExample],
    tokenizer: BPETokenizer,
    batch_size: int,
    max_len: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TranslationDataset(examples, tokenizer, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )
