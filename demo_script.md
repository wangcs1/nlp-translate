# 1-2 分钟程序演示脚本

## 画面 1：项目结构

打开项目目录，说明本项目实现英文到中文机器翻译，核心代码在 `mt_transformer/`：

- `tokenizer.py`：joint BPE 子词分词
- `model.py`：Encoder-Decoder Transformer
- `train.py`：训练入口
- `translate.py`：翻译推理
- `evaluate.py`：BLEU 评估

## 画面 2：数据集

展示两份数据：

- `data/sample_en_zh.tsv`：小型样例，用于快速跑通。
- `data/en_zh_10k.tsv`：12,000 条教学平行语料，用于更完整的训练展示。

说明 TSV 格式为：

```text
english sentence<TAB>中文句子
```

## 画面 3：训练 fancy Transformer

运行：

```bash
python -m mt_transformer.train --data data/en_zh_10k.tsv --preset fancy --epochs 40 --batch-size 32 --amp
```

讲解要点：模型使用多头注意力、GELU 前馈网络、joint BPE、共享 embedding、标签平滑、Noam 学习率、早停和梯度裁剪。训练过程中会保存验证集 loss 最低的 checkpoint。

## 画面 4：翻译推理

运行：

```bash
python -m mt_transformer.translate --sentence "attention mechanism helps the model focus on important words ."
```

讲解要点：推理阶段默认使用 Beam Search，同时保留多个候选翻译，并结合长度惩罚选择最终中文输出。

## 画面 5：实验评估

运行：

```bash
python -m mt_transformer.evaluate --data data/en_zh_10k.tsv
```

展示 BLEU 分数和 `outputs/translations.txt`。最后说明：模板生成语料适合课程展示和流程验证，真实翻译效果应继续加入 OPUS、TED、新闻等真实英中平行语料。
