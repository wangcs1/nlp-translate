# 基于 Transformer 的英文到中文机器翻译

这是一个课程实践用的完整小项目，核心模型为 Encoder-Decoder Transformer，支持联合 BPE 子词分词、位置编码、多头注意力、Label Smoothing、Noam 学习率调度、梯度裁剪、Beam Search 解码和 BLEU 评估。

## 目录

- `mt_transformer/tokenizer.py`：离线可训练的 joint BPE tokenizer
- `mt_transformer/model.py`：Transformer 翻译模型
- `mt_transformer/train.py`：训练脚本
- `mt_transformer/translate.py`：模型推理脚本
- `mt_transformer/evaluate.py`：BLEU 评估与结果导出
- `mt_transformer/pretrained_translate.py`：可选预训练 Transformer 推理增强
- `data/sample_en_zh.tsv`：可直接跑通的英中样例数据
- `report.md`：课程报告正文

## 安装依赖

```bash
pip install -r requirements.txt
```

如果只跑自研 Transformer，最低需要 `torch` 和 `tqdm`。`sacrebleu` 用于评估，`transformers` 和 `sentencepiece` 只用于可选预训练推理脚本。

## 训练

```bash
python -m mt_transformer.train --data data/sample_en_zh.tsv --epochs 60
```

训练完成后会生成：

- `checkpoints/transformer_en_zh/best.pt`
- `checkpoints/transformer_en_zh/tokenizer.json`

真实作业展示时建议把 `data/sample_en_zh.tsv` 替换为更大的英中平行语料，例如自建课程数据、新闻翻译数据或 OPUS/TED 风格数据。格式保持每行 `英文<TAB>中文` 即可。

## 翻译演示

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

使用贪心解码：

```bash
python -m mt_transformer.translate --greedy --sentence "machine translation converts one language into another language ."
```

## 评估

```bash
python -m mt_transformer.evaluate --data data/sample_en_zh.tsv
```

程序会输出 BLEU 分数，并把样例翻译保存到 `outputs/translations.txt`。

## 可选：预训练 Transformer 高质量推理

如果机器可以联网或已经缓存 Hugging Face 模型，可以运行：

```bash
python -m mt_transformer.pretrained_translate --sentence "Transformer has greatly improved machine translation quality."
```

这部分用于演示工业级 Transformer 迁移能力；课程代码的核心实现仍然是本项目中的自研 Transformer。
