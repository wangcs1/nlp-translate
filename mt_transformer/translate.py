from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .decode import beam_translate, greedy_translate
from .model import TransformerTranslator
from .tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English into Chinese.")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_en_zh/best.pt")
    parser.add_argument("--tokenizer", default="checkpoints/transformer_en_zh/tokenizer.json")
    parser.add_argument("--sentence", default="attention mechanism helps the model focus on important words .")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of beam search.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    if args.greedy:
        translation = greedy_translate(model, tokenizer, args.sentence, device, max_len=args.max_len)
    else:
        translation = beam_translate(
            model,
            tokenizer,
            args.sentence,
            device,
            beam_size=args.beam_size,
            max_len=args.max_len,
        )
    print(f"EN: {args.sentence}")
    print(f"ZH: {translation}")


def load_model(checkpoint_path: str | Path, tokenizer_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = BPETokenizer.load(tokenizer_path)
    config = checkpoint["config"]
    model = TransformerTranslator(
        vocab_size=checkpoint.get("vocab_size", len(tokenizer.itos)),
        pad_id=checkpoint.get("pad_id", tokenizer.pad_id),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["layers"],
        num_decoder_layers=config["layers"],
        dim_feedforward=config["ffn_dim"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        share_embeddings=config.get("share_embeddings", False),
        share_decoder_generator=config.get("share_decoder_generator", False),
        learned_positional=config.get("learned_positional", False),
        activation=config.get("activation", "swiglu"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    main()
