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


MODEL_PRESETS = {
    "tiny": {"d_model": 192, "nhead": 6, "layers": 3, "ffn_dim": 768, "dropout": 0.15},
    "small": {"d_model": 256, "nhead": 8, "layers": 4, "ffn_dim": 1024, "dropout": 0.15},
    "base": {"d_model": 512, "nhead": 8, "layers": 6, "ffn_dim": 2048, "dropout": 0.1},
    "fancy": {"d_model": 512, "nhead": 8, "layers": 8, "ffn_dim": 2048, "dropout": 0.1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer EN-ZH translator.")
    parser.add_argument("--data", default="data/sample_en_zh.tsv", help="TSV file: English<TAB>Chinese.")
    parser.add_argument("--extra-data", nargs="*", default=[], help="Optional extra TSV files to merge.")
    parser.add_argument("--save-dir", default="checkpoints/transformer_en_zh")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="small")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--ffn-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--share-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--share-decoder-generator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learned-positional", action="store_true")
    parser.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    apply_model_preset(args)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data, args.extra_data)
    train_examples, valid_examples = split_examples(examples, seed=args.seed)
    print(f"loaded {len(examples)} sentence pairs: {len(train_examples)} train / {len(valid_examples)} valid")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokenizer.train([x.src for x in train_examples] + [x.tgt for x in train_examples])
    tokenizer.save(save_dir / "tokenizer.json")

    train_loader = build_loader(
        train_examples,
        tokenizer,
        args.batch_size,
        args.max_len,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_loader(
        valid_examples,
        tokenizer,
        args.batch_size,
        args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
    )

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
        share_embeddings=args.share_embeddings,
        share_decoder_generator=args.share_decoder_generator,
        learned_positional=args.learned_positional,
        activation=args.activation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_valid = math.inf
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            scheduler,
            device,
            train=True,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            amp=args.amp,
        )
        valid_loss = run_epoch(
            model,
            valid_loader,
            criterion,
            scheduler,
            device,
            train=False,
            grad_accum_steps=1,
            scaler=scaler,
            amp=False,
        )
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            bad_epochs = 0
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
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping after {args.patience} epochs without validation improvement")
                break


def run_epoch(
    model,
    loader,
    criterion,
    scheduler,
    device,
    train: bool,
    grad_accum_steps: int,
    scaler,
    amp: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, leave=False, desc="train" if train else "valid")
    if train:
        scheduler.zero_grad()
    for step, (src, tgt) in enumerate(iterator, start=1):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if train:
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                if step % grad_accum_steps == 0 or step == len(loader):
                    scaler.unscale_(scheduler.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scheduler.advance()
                    scaler.step(scheduler.optimizer)
                    scaler.update()
                    scheduler.zero_grad()

        tokens = tgt_out.ne(model.pad_id).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        iterator.set_postfix(loss=loss.item())
    return total_loss / max(1, total_tokens)


def apply_model_preset(args: argparse.Namespace) -> None:
    preset = MODEL_PRESETS[args.preset]
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def load_examples(data_path: str, extra_paths: list[str]):
    seen: set[tuple[str, str]] = set()
    examples = []
    for path in [data_path, *extra_paths]:
        for example in read_parallel_tsv(path):
            key = (example.src, example.tgt)
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
    if len(examples) < 4:
        raise ValueError("Need at least 4 sentence pairs for train/validation split.")
    return examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
