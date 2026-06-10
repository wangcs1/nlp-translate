from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .data import build_loader, read_parallel_tsv, split_examples
from .model import TransformerTranslator
from .optim import NoamScheduler
from .tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer EN-ZH translator.")
    parser.add_argument("--data", default="data/sample_en_zh.tsv", help="TSV file: English<TAB>Chinese.")
    parser.add_argument("--save-dir", default="checkpoints/transformer_en_zh")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=1200)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    examples = read_parallel_tsv(args.data)
    train_examples, valid_examples = split_examples(examples, seed=args.seed)
    tokenizer = BPETokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.train([x.src for x in train_examples] + [x.tgt for x in train_examples])
    tokenizer.save(save_dir / "tokenizer.json")

    train_loader = build_loader(train_examples, tokenizer, args.batch_size, args.max_len, shuffle=True)
    valid_loader = build_loader(valid_examples, tokenizer, args.batch_size, args.max_len, shuffle=False)

    model = TransformerTranslator(
        vocab_size=len(tokenizer.itos),
        pad_id=tokenizer.pad_id,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        max_len=args.max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing)

    best_valid = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, scheduler, device, train=True)
        valid_loss = run_epoch(model, valid_loader, criterion, scheduler, device, train=False)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "vocab_size": len(tokenizer.itos),
                    "pad_id": tokenizer.pad_id,
                },
                save_dir / "best.pt",
            )
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


def run_epoch(model, loader, criterion, scheduler, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "valid")
    for src, tgt in iterator:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.set_grad_enabled(train):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if train:
                scheduler.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()

        tokens = tgt_out.ne(model.pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        iterator.set_postfix(loss=loss.item())
    return total_loss / max(1, total_tokens)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
