# 基于 Transformer 的英中机器翻译课程报告

## 1. 任务目标

本项目实现一个英文到中文的神经机器翻译系统。输入英文句子后，模型生成对应中文译文。项目包含数据处理、BPE 分词、Transformer 训练、Beam Search 推理和 BLEU 评估完整流程。

## 2. 数据集扩展

项目提供两类数据：

- `data/sample_en_zh.tsv`：小型人工整理样例，用于快速验证流程。
- `data/en_zh_10k.tsv`：通过 `scripts/generate_parallel_corpus.py` 生成的 12,000 条教学平行语料。

所有数据均为 TSV 格式：

```text
english sentence<TAB>中文句子
```

10k 级语料通过多个模板、主语、谓语、宾语、领域名词、形容词、地点和副词组合生成，覆盖机器翻译、自然语言处理、训练、评估、数据清洗、模型优化等课程相关主题。它适合课程展示、训练流程验证和模型容量实验。若追求真实翻译性能，应继续加入 OPUS、TED、新闻或技术文档等真实英中平行语料。

## 3. 模型结构

模型采用 Encoder-Decoder Transformer：

- Encoder 读取英文 token 序列。
- Decoder 自回归生成中文 token。
- Cross Attention 连接源语言表示和目标语言生成过程。
- 输出层预测目标词表分布。

当前实现支持以下增强：

- joint BPE 子词词表，降低未知词比例。
- 源端和目标端 embedding 权重共享，适配 joint BPE。
- decoder embedding 与 generator 权重共享，减少参数并提升泛化。
- 可选学习式位置编码。
- GELU 前馈激活函数。
- Pre-LN Transformer，提高深层训练稳定性。
- Label Smoothing、Noam 学习率、梯度裁剪、梯度累积、早停和可选 AMP。

## 4. Fancy Transformer 配置

训练脚本提供多个预设：

- `tiny`：轻量调试配置。
- `small`：默认推荐配置。
- `base`：接近 Transformer Base 的 6 层、512 维配置。
- `fancy`：8 层、512 hidden、8 heads、2048 FFN、GELU、权重共享，适合 10k+ 语料实验。

推荐命令：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --preset fancy --epochs 40 --batch-size 32 --amp
```

显存不足时可降低 batch size 并启用梯度累积：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --preset fancy --batch-size 8 --grad-accum-steps 4
```

## 5. 推理与评估

推理默认使用 Beam Search：

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

评估使用 BLEU：

```bash
python -m mt_transformer.evaluate --data data/en_zh_10k.tsv
```

评估脚本会输出 BLEU，并保存翻译结果到 `outputs/translations.txt`。

## 6. 总结

优化后的项目从最初的小样例 Transformer 翻译系统，扩展为支持 10k 级语料训练和多档模型配置的完整实验项目。它既能用于课堂演示，也能作为继续接入真实平行语料、扩大模型规模和改进翻译质量的基础。
