from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import read_parallel_tsv
from .decode import beam_translate
from .translate import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EN-ZH translator with BLEU.")
    parser.add_argument("--data", default="data/en_zh_quality.tsv")
    parser.add_argument("--checkpoint", default="checkpoints/transformer_en_zh/best.pt")
    parser.add_argument("--tokenizer", default="checkpoints/transformer_en_zh/tokenizer.json")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    examples = read_parallel_tsv(args.data)
    if args.limit:
        examples = examples[: args.limit]
    hypotheses = [
        beam_translate(model, tokenizer, ex.src, device, beam_size=args.beam_size, max_len=args.max_len)
        for ex in examples
    ]
    references = [ex.tgt for ex in examples]

    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="zh")
        print(f"BLEU = {bleu.score:.2f}")
    except Exception as exc:
        print(f"sacrebleu unavailable ({exc}); printing translations only.")

    out_path = Path("outputs/translations.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex, hyp in zip(examples, hypotheses):
            f.write(f"EN: {ex.src}\nREF: {ex.tgt}\nHYP: {hyp}\n\n")
    print(f"Saved translations to {out_path}")


if __name__ == "__main__":
    main()
