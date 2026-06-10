# 1-2 分钟程序演示视频脚本

## 画面 1：项目结构

打开项目目录，说明本项目实现英文到中文机器翻译，核心代码在 `mt_transformer/`：

- `tokenizer.py`：BPE 子词分词
- `model.py`：Transformer Encoder-Decoder
- `train.py`：训练入口
- `translate.py`：翻译推理
- `evaluate.py`：BLEU 评估

## 画面 2：训练模型

运行：

```bash
python -m mt_transformer.train --data data/sample_en_zh.tsv --epochs 60
```

讲解要点：模型使用多头注意力、位置编码、Label Smoothing 和 Noam 学习率调度，训练过程中会保存验证集 loss 最低的 checkpoint。

## 画面 3：翻译推理

运行：

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

讲解要点：推理阶段使用 Beam Search，同时保留多个候选翻译，选择综合概率和长度惩罚后最优的中文句子。

## 画面 4：实验评估

运行：

```bash
python -m mt_transformer.evaluate --data data/sample_en_zh.tsv
```

展示 BLEU 分数和 `outputs/translations.txt`。最后说明：样例数据用于跑通流程，实际提升效果需要替换为更大规模英中平行语料。
