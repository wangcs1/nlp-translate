# EN-ZH Transformer Translator

This project keeps one clean training path for English-to-Chinese machine translation:

1. Build a filtered high-quality TSV corpus from a public parallel dataset.
2. Train one strong Transformer encoder-decoder model.
3. Translate and evaluate from the saved checkpoint.

The training data format is fixed:

```text
english sentence<TAB>中文句子
```

## Install

```bash
pip install -r requirements.txt
```

## Build Data

The default builder downloads curated OPUS corpora with more formal written Chinese:
`News-Commentary` and `WMT-News`. It keeps cleaner sentence pairs by applying
language, length, ratio, markup, URL, duplicate, and punctuation filters.

```bash
python scripts/build_quality_data.py --max-examples 30000 --output data/en_zh_quality.tsv
```

For a smaller first run:

```bash
python scripts/build_quality_data.py --max-examples 5000 --output data/en_zh_quality.tsv
```

## Train

```bash
python -m mt_transformer.train --amp
```

The only model path is the strong Transformer configuration in `mt_transformer/train.py`:

- 8 encoder layers and 8 decoder layers
- 512 hidden size
- 8 attention heads
- 2048 feed-forward dimension
- GELU activation
- learned positional embeddings
- shared source/target embeddings
- shared decoder embedding/output projection
- label smoothing, AdamW, Noam learning-rate schedule, gradient clipping, early stopping

Artifacts are written to:

```text
checkpoints/transformer_en_zh/best.pt
checkpoints/transformer_en_zh/tokenizer.json
```

## Translate

```bash
python -m mt_transformer.translate --sentence "Transformer models learn from parallel data."
```

## Evaluate

```bash
python -m mt_transformer.evaluate --data data/en_zh_quality.tsv
```

Evaluation writes examples to `outputs/translations.txt`.
