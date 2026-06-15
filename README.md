# 基于 Transformer 的英中机器翻译

这是一个可运行的课程项目，核心是 Encoder-Decoder Transformer，支持 joint BPE、位置编码、标签平滑、Noam 学习率、梯度裁剪、Beam Search 和 BLEU 评估。

## 项目结构

- `mt_transformer/tokenizer.py`：离线 joint BPE tokenizer
- `mt_transformer/model.py`：支持权重共享和 fancy 配置的 Transformer
- `mt_transformer/train.py`：训练脚本
- `mt_transformer/translate.py`：推理脚本
- `mt_transformer/evaluate.py`：BLEU 评估
- `mt_transformer/pretrained_translate.py`：可选预训练模型推理
- `data/sample_en_zh.tsv`：小型 smoke test 数据
- `data/en_zh_10k.tsv`：12k 条可复现教学平行语料
- `scripts/generate_parallel_corpus.py`：生成 10k+ 语料的脚本

## 安装

```bash
pip install -r requirements.txt
```

## 数据

默认提供两份数据：

- `data/sample_en_zh.tsv`：约百条人工整理句对，适合快速检查代码能否跑通。
- `data/en_zh_10k.tsv`：12,000 条模板组合生成的教学平行语料，适合训练流程展示和模型容量实验。

重新生成 10k+ 数据：

```bash
python scripts/generate_parallel_corpus.py --output data/en_zh_10k.tsv --size 12000
```

真实翻译质量仍然主要取决于真实平行语料。你可以把 OPUS、TED、新闻或课程语料整理成 `英文<TAB>中文` 格式，再通过 `--extra-data` 合并训练。

## 使用外部优质语料

推荐优先找已经整理好的英中平行语料：

- OPUS：开放平行语料集合，常用子库包括 OpenSubtitles、TED2020、News-Commentary、WikiMatrix、CCAligned 等。
- Tatoeba：句子级多语言平行语料，规模较小但质量较干净。
- WMT Chinese-English：机器翻译比赛常用数据，质量较好，但需要注意具体年份和授权说明。
- Hugging Face Datasets：很多数据集可以直接下载为 `translation` 字段，再导出 TSV。

本项目训练格式固定为：

```text
english sentence<TAB>中文句子
```

如果外部数据已经是 TSV：

```bash
python scripts/prepare_external_parallel.py --input-tsv raw/external.tsv --output data/external_clean.tsv
```

如果外部数据是两个对齐文件，例如 `train.en` 和 `train.zh`：

```bash
python scripts/prepare_external_parallel.py --src-file raw/train.en --tgt-file raw/train.zh --output data/external_clean.tsv
```

如果外部数据来自 Hugging Face `translation` 数据集：

```bash
python scripts/download_hf_parallel.py --dataset Helsinki-NLP/opus-100 --config en-zh --split train --output data/opus100_en_zh.tsv --shuffle --max-examples 200000
```

如果你有别的 Hugging Face 数据集，只要它返回 `translation` 字段，通常都能这样导出。

抽样一部分先试训：

```bash
python scripts/prepare_external_parallel.py --src-file raw/train.en --tgt-file raw/train.zh --output data/external_100k.tsv --limit 100000
```

合并训练：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --extra-data data/external_clean.tsv --preset fancy --epochs 40 --batch-size 32 --amp
```

如果外部真实语料足够大，也可以只训练外部语料：

```bash
python -m mt_transformer.train --data data/external_clean.tsv --preset base --vocab-size 16000 --epochs 30 --batch-size 32 --amp
```

## 训练

小样例训练：

```bash
python -m mt_transformer.train --data data/sample_en_zh.tsv --preset small
```

10k+ fancy Transformer 训练：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --preset fancy --epochs 40 --batch-size 32 --amp
```

合并外部语料：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --extra-data path/to/opus.tsv path/to/ted.tsv --preset fancy
```

训练产物：

- `checkpoints/transformer_en_zh/best.pt`
- `checkpoints/transformer_en_zh/tokenizer.json`

## 模型配置

预设：

- `tiny`：原始轻量配置，适合 CPU 快速跑通。
- `small`：默认推荐，小数据也比较稳。
- `base`：6 层、512 hidden 的 Transformer Base 风格配置。
- `fancy`：8 层、GELU、共享 embedding/generator，适合 10k+ 或更大语料展示。

重要训练增强：

- `--share-embeddings`：共享源/目标词嵌入，适合 joint BPE。
- `--share-decoder-generator`：共享目标嵌入和输出投影权重，减少参数并提升泛化。
- `--learned-positional`：使用可学习位置编码。
- `--grad-accum-steps`：梯度累积，显存不足时模拟更大 batch。
- `--patience`：早停，减少过拟合。
- `--amp`：CUDA 混合精度训练。

## 推理

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

使用贪心解码：

```bash
python -m mt_transformer.translate --greedy --sentence "machine translation converts one language into another language ."
```

## 评估

```bash
python -m mt_transformer.evaluate --data data/en_zh_10k.tsv
```

评估结果会输出 BLEU，并把样例保存到 `outputs/translations.txt`。

## 可选：预训练 Transformer 推理

如果机器可以联网或已经缓存 Hugging Face 模型：

```bash
python -m mt_transformer.pretrained_translate --sentence "Transformer has greatly improved machine translation quality."
```

这部分用于展示工业级预训练 Transformer 的迁移能力；项目核心实现仍然是本仓库里的自研 Transformer。
