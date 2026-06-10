from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional high-quality pretrained Transformer inference.")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-zh")
    parser.add_argument("--sentence", default="Transformer is a powerful neural network architecture.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Please install transformers and sentencepiece first: pip install -r requirements.txt") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    inputs = tokenizer(args.sentence, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=5, max_new_tokens=96)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
