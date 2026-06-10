# 基于 Transformer 的英文到中文机器翻译课程报告

## 1. 任务目标

本项目实现一个英文到中文的神经机器翻译系统。输入英文句子，模型自动生成对应中文译文。项目提交内容包括可运行源代码、训练与推理脚本、实验评估流程和演示说明。

## 2. 方法概述

系统采用 Encoder-Decoder Transformer 架构。源语言英文和目标语言中文使用联合 BPE 子词词表，使模型可以同时处理英文单词、中文字符、标点符号和低频词片段。训练阶段采用 Teacher Forcing；推理阶段采用 Beam Search，保留多个候选翻译路径，从而提升生成质量。

核心技术包括：

- Joint BPE 子词分词：缓解未登录词问题，提升小数据集泛化能力。
- 多头自注意力：从不同表示子空间捕获长距离依赖。
- 位置编码：向无循环结构的 Transformer 注入词序信息。
- Label Smoothing：降低模型过拟合和过度自信。
- Noam 学习率调度：先 warmup 再衰减，符合经典 Transformer 训练策略。
- Beam Search + Length Penalty：在翻译流畅度和长度之间取得平衡。
- BLEU 自动评估：量化模型输出与参考译文的接近程度。

## 3. 模型结构

模型由编码器、解码器和输出生成层组成。

编码器接收英文 token 序列，通过词嵌入、位置编码和多层 self-attention 提取上下文表示。解码器在已生成中文 token 的基础上，通过 masked self-attention 保证自回归生成，再通过 cross-attention 读取编码器输出。最终线性层将隐藏状态映射到词表概率分布。

模型默认配置为：

- `d_model=192`
- `nhead=6`
- 编码器层数 `3`
- 解码器层数 `3`
- FFN 隐层维度 `768`
- Dropout `0.15`

在更大数据集和 GPU 环境下，可提升到 `d_model=512`、`nhead=8`、`layers=6`，更接近论文级 Transformer Base 配置。

## 4. 数据处理

输入数据格式为 TSV，每行包含一个英中句对：

```text
english sentence<TAB>中文句子
```

项目提供 `data/sample_en_zh.tsv` 作为可运行样例。实际实验可替换为更大规模英中平行语料。训练脚本会自动划分训练集和验证集，并只使用训练集学习 BPE 词表，避免验证集信息泄露。

## 5. 训练过程

训练脚本为：

```bash
python -m mt_transformer.train --data data/sample_en_zh.tsv --epochs 60
```

训练流程：

1. 读取英中平行语料。
2. 训练 joint BPE tokenizer。
3. 构造 batch，并对不同长度序列进行 padding。
4. 使用交叉熵损失和 label smoothing 训练模型。
5. 使用 Noam schedule 动态调整学习率。
6. 在验证集 loss 最低时保存最优 checkpoint。

## 6. 推理与演示

训练完成后，可运行：

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

程序输出英文原句和中文译文。演示视频可录制以下内容：

1. 展示项目目录与核心代码。
2. 运行训练命令，展示 loss 下降。
3. 运行翻译命令，输入 2 到 3 个英文句子。
4. 运行评估命令，展示 BLEU 分数和 `outputs/translations.txt`。

## 7. 实验结果与分析

在样例小数据集上，模型主要用于验证流程可运行。由于训练语料较少，模型会明显记忆训练样本，泛化能力有限。扩大训练语料后，BPE 能显著降低 OOV 问题，Beam Search 能让译文更完整，Label Smoothing 可以缓解过拟合。

影响翻译质量的关键因素：

- 数据规模和句对质量是最主要因素。
- 词表大小需要与语料规模匹配，小语料使用较小词表更稳定。
- Beam Size 通常设置为 4 或 5，过大会增加推理时间。
- 模型层数和隐藏维度越大，表达能力越强，但更依赖 GPU 和数据量。

## 8. 创新点

相比只调用现成库的简单实现，本项目包含完整 Transformer 训练闭环，并加入多项经典机器翻译增强技术：

- 自实现 BPE 子词算法，减少外部依赖。
- 使用 Transformer 原论文的 Noam 学习率策略。
- 使用 Label Smoothing 提升泛化。
- 使用 Beam Search 和长度惩罚改善生成质量。
- 提供可选预训练 Transformer 脚本，用于展示工业级模型效果。

## 9. 总结

本项目完成了英文到中文机器翻译系统的设计、实现、训练、推理和评估。系统结构清晰，代码可运行，便于课堂演示和后续扩展。若继续改进，可从更大规模语料训练、引入 Transformer Base 配置、使用混合精度训练和加载预训练权重等方向提升翻译效果。
