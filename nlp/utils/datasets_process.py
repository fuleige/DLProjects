import os
import json
import random
from collections import defaultdict


def preprocess_dataset(input_path, train_ratio=0.8, val_ratio=0.2):
    """
    处理文本分类数据集, 将其转换为jsonl格式, 并划分为训练集, 验证集和测试集
    """
    # 1. 读取原始数据集
    class_datas = defaultdict(list)
    with open(input_path, 'r', encoding='utf-8') as infile:
        infile.readline()  # 跳过标题行
        for line in infile:
            label, text = line.strip().split(',', maxsplit=1)
            class_datas[int(label)].append(json.dumps(
                {"text": text, "label": int(label)}, ensure_ascii=False))
    # 2. 划分数据集
    datasets = {"train": [], "val": [], "test": []}
    for label, texts in class_datas.items():
        random.shuffle(texts)
        total = len(texts)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        datasets["train"].extend(texts[:train_end])
        datasets["val"].extend(texts[train_end:val_end])
        datasets["test"].extend(texts[val_end:])
    # 3. 写入jsonl文件
    for name, data in datasets.items():
        # output_path = os.path.splitext(input_path)[0] + f"_{name}.jsonl"
        output_path = os.path.join(
            os.path.dirname(input_path), f"{name}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in data:
                outfile.write(line + '\n')


def main():
    raw_csv_path = '/root/codes/nlp/text_classification/datasets/waimai_10k.csv'
    preprocess_dataset(raw_csv_path)


if __name__ == "__main__":
    main()
