# EN-ZH Transformer Translator

This project keeps one reproducible English-to-Chinese training path:

1. Build a filtered TSV corpus from curated public parallel data.
2. Train one Transformer encoder-decoder model.
3. Translate and evaluate from the saved checkpoint.

The training data format is:

```text
english sentence<TAB>中文句子
```

## Install

```bash
pip install -r requirements.txt
```

## Build Data

The default builder downloads and mixes several OPUS sources: `News-Commentary`,
`WMT-News`, `WikiMatrix`, `MultiUN`, and a small `QED` slice. It also adds a
small in-domain machine-translation seed set. The script caches raw zip files in
`data/raw_opus` so repeated builds are faster.

```bash
python scripts/build_quality_data.py --max-examples 180000 --output data/en_zh_quality.tsv
```

## Train

```bash
python -m mt_transformer.train --amp
```

To train longer:

```bash
python -m mt_transformer.train --amp --epochs 120
```

Training always runs for the requested number of epochs. There is no early
stopping. The trainer writes both the latest checkpoint and the best validation
checkpoint.

The current model is a modernized encoder-decoder Transformer:

- SentencePiece unigram subword tokenizer with byte fallback
- learned positional embeddings
- Pre-Norm encoder and decoder blocks
- RMSNorm instead of LayerNorm
- SwiGLU feed-forward networks
- scaled residual branches for deeper training stability
- shared source/target embeddings
- shared decoder embedding/output projection
- AdamW, Noam learning-rate schedule, label smoothing, and gradient clipping

Artifacts are written to:

```text
checkpoints/transformer_en_zh/best.pt
checkpoints/transformer_en_zh/last.pt
checkpoints/transformer_en_zh/tokenizer.json
```

## Translate

News-domain examples are closest to the main training data:

```bash
python -m mt_transformer.translate --sentence "The global economy is facing serious challenges."
```

After rebuilding data and retraining with the technical seed set, this project
demo sentence should also be covered:

```bash
python -m mt_transformer.translate --sentence "Transformer models learn from parallel data."
```

## Evaluate

```bash
python -m mt_transformer.evaluate --data data/en_zh_quality.tsv --limit 500
```

Full beam-search evaluation is much slower:

```bash
python -m mt_transformer.evaluate --data data/en_zh_quality.tsv
```

Older checkpoints trained with the previous `nn.Transformer` architecture are
not compatible with the current model. Rebuild the data and retrain after this
update.
