from __future__ import annotations

import argparse
import random
from pathlib import Path


SUBJECTS = [
    ("i", "我"),
    ("you", "你"),
    ("we", "我们"),
    ("they", "他们"),
    ("the student", "学生"),
    ("the teacher", "老师"),
    ("the engineer", "工程师"),
    ("the researcher", "研究员"),
    ("the translator", "译者"),
    ("the model", "模型"),
    ("the tokenizer", "分词器"),
    ("the encoder", "编码器"),
    ("the decoder", "解码器"),
    ("the system", "系统"),
    ("the program", "程序"),
    ("the dataset", "数据集"),
    ("the report", "报告"),
    ("the experiment", "实验"),
    ("the application", "应用"),
    ("the company", "公司"),
]

VERBS = [
    ("study", "学习"),
    ("analyze", "分析"),
    ("translate", "翻译"),
    ("evaluate", "评估"),
    ("improve", "改进"),
    ("debug", "调试"),
    ("train", "训练"),
    ("test", "测试"),
    ("compare", "比较"),
    ("summarize", "总结"),
    ("build", "构建"),
    ("optimize", "优化"),
    ("read", "读取"),
    ("write", "编写"),
    ("save", "保存"),
    ("load", "加载"),
    ("generate", "生成"),
    ("clean", "清洗"),
    ("align", "对齐"),
    ("review", "检查"),
]

OBJECTS = [
    ("a sentence", "一个句子"),
    ("the text", "文本"),
    ("the code", "代码"),
    ("the translation", "译文"),
    ("the vocabulary", "词表"),
    ("the checkpoint", "检查点"),
    ("the training data", "训练数据"),
    ("the validation set", "验证集"),
    ("the model output", "模型输出"),
    ("the attention weights", "注意力权重"),
    ("the experiment result", "实验结果"),
    ("the bleu score", "BLEU分数"),
    ("the source sentence", "源句子"),
    ("the target sentence", "目标句子"),
    ("the neural network", "神经网络"),
    ("the transformer architecture", "Transformer架构"),
    ("the language pair", "语言对"),
    ("the corpus", "语料库"),
    ("the report", "报告"),
    ("the demo", "演示"),
]

ADVERBS = [
    ("carefully", "认真地"),
    ("quickly", "快速地"),
    ("slowly", "慢慢地"),
    ("accurately", "准确地"),
    ("automatically", "自动地"),
    ("step by step", "逐步地"),
    ("with beam search", "使用束搜索"),
    ("with more data", "使用更多数据"),
    ("during training", "在训练期间"),
    ("after evaluation", "在评估之后"),
]

NOUNS = [
    ("machine translation", "机器翻译"),
    ("natural language processing", "自然语言处理"),
    ("deep learning", "深度学习"),
    ("attention mechanism", "注意力机制"),
    ("positional encoding", "位置编码"),
    ("subword tokenization", "子词分词"),
    ("gradient accumulation", "梯度累积"),
    ("early stopping", "早停"),
    ("label smoothing", "标签平滑"),
    ("mixed precision", "混合精度"),
    ("training stability", "训练稳定性"),
    ("data quality", "数据质量"),
    ("translation fluency", "翻译流畅度"),
    ("model capacity", "模型容量"),
    ("parallel corpus", "平行语料"),
    ("human evaluation", "人工评估"),
]

ADJECTIVES = [
    ("important", "重要"),
    ("useful", "有用"),
    ("stable", "稳定"),
    ("accurate", "准确"),
    ("efficient", "高效"),
    ("powerful", "强大"),
    ("clean", "干净"),
    ("diverse", "多样"),
    ("reliable", "可靠"),
    ("interesting", "有趣"),
    ("modern", "现代"),
    ("practical", "实用"),
]

PLACES = [
    ("in the classroom", "在教室里"),
    ("in the laboratory", "在实验室里"),
    ("on the server", "在服务器上"),
    ("on the gpu", "在GPU上"),
    ("in the project", "在项目中"),
    ("in the report", "在报告中"),
    ("during the course", "在课程中"),
    ("for the demo", "为了演示"),
]

QUESTION_WORDS = [
    ("what", "什么"),
    ("why", "为什么"),
    ("how", "如何"),
    ("when", "什么时候"),
    ("where", "在哪里"),
]

BASE_PAIRS = [
    ("hello .", "你好。"),
    ("how are you ?", "你好吗？"),
    ("thank you very much .", "非常感谢你。"),
    ("machine translation converts one language into another language .", "机器翻译把一种语言转换成另一种语言。"),
    ("attention mechanism helps the model focus on important words .", "注意力机制帮助模型关注重要的词。"),
    ("large datasets usually produce better translation results .", "大规模数据集通常会产生更好的翻译结果。"),
]


def make_templates() -> list:
    return [
        lambda s, v, o, a, n, adj, p, q: (
            f"{s[0]} {v[0]} {o[0]} {a[0]} .",
            f"{s[1]}{a[1]}{v[1]}{o[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"{s[0]} {v[0]} {o[0]} {p[0]} .",
            f"{s[1]}{p[1]}{v[1]}{o[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"{n[0]} is {adj[0]} for {o[0]} .",
            f"{n[1]}对{o[1]}很{adj[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"the {adj[0]} {n[0]} improves {o[0]} .",
            f"{adj[1]}的{n[1]}改进{o[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"we use {n[0]} to {v[0]} {o[0]} .",
            f"我们使用{n[1]}来{v[1]}{o[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"please {v[0]} {o[0]} {a[0]} .",
            f"请{a[1]}{v[1]}{o[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"can {s[0]} {v[0]} {o[0]} ?",
            f"{s[1]}能{v[1]}{o[1]}吗？",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"{q[0]} does {s[0]} {v[0]} ?",
            f"{s[1]}{q[1]}{v[1]}？",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"{s[0]} needs {adj[0]} {n[0]} .",
            f"{s[1]}需要{adj[1]}的{n[1]}。",
        ),
        lambda s, v, o, a, n, adj, p, q: (
            f"{n[0]} makes {o[0]} more {adj[0]} .",
            f"{n[1]}让{o[1]}更加{adj[1]}。",
        ),
    ]


def generate_pairs(target_size: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    templates = make_templates()
    pairs = list(BASE_PAIRS)
    seen = set(pairs)

    attempts = 0
    while len(pairs) < target_size:
        attempts += 1
        if attempts > target_size * 50:
            raise RuntimeError("Could not generate enough unique sentence pairs.")

        template = rng.choice(templates)
        pair = template(
            rng.choice(SUBJECTS),
            rng.choice(VERBS),
            rng.choice(OBJECTS),
            rng.choice(ADVERBS),
            rng.choice(NOUNS),
            rng.choice(ADJECTIVES),
            rng.choice(PLACES),
            rng.choice(QUESTION_WORDS),
        )
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible EN-ZH teaching corpus.")
    parser.add_argument("--output", default="data/en_zh_10k.tsv")
    parser.add_argument("--size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pairs = generate_pairs(args.size, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(f"{src}\t{tgt}" for src, tgt in pairs) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(pairs)} sentence pairs to {output}")


if __name__ == "__main__":
    main()
