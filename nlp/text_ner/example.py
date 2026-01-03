from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
import torch
import os
from pathlib import Path
import json
from seqeval.metrics import precision_score, recall_score, f1_score

"""
数据格式：
    字符 \t 标签
    空行分隔不同句子

BIO格式示例：
    们      O
    收      O
    藏      O
    北      B-LOC
    京      I-LOC
    史      O

BIOES格式示例（推荐）：
    我      O
    在      O
    北      B-LOC
    京      I-LOC
    市      E-LOC
    工      O
    作      O

BIOES标签说明：
    B (Begin): 实体开始
    I (Inside): 实体内部
    O (Outside): 非实体
    E (End): 实体结束
    S (Single): 单字符实体
"""


def load_bio_format_data(file_path):
    sentences = []
    sentence_tokens = []
    sentence_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                # 空行表示一个句子结束
                if sentence_tokens:
                    sentences.append({
                        'tokens': sentence_tokens,
                        'ner_tags': sentence_labels
                    })
                    sentence_tokens = []
                    sentence_labels = []
            else:
                # 处理每一行：字符\t标签
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[-1]  # 最后一列是标签
                    sentence_tokens.append(token)
                    sentence_labels.append(label)

    # 处理最后一个句子
    if sentence_tokens:
        sentences.append({
            'tokens': sentence_tokens,
            'ner_tags': sentence_labels
        })

    return sentences


def get_label_list(data_list):
    """从数据中提取所有唯一的标签"""
    labels = set()
    for item in data_list:
        labels.update(item['ner_tags'])
    return sorted(list(labels))


"""
将标签与分词后的token对齐
BERT分词器可能会将一个字拆分成多个子词(subword)，需要处理标签对齐

策略：
- 对于被拆分的token，第一个subword保持原标签
- 后续的subword使用-100（在loss计算中会被忽略）
- 特殊token（[CLS], [SEP]等）也使用-100

对于BIOES格式的特殊处理：
- B-XXX的后续subword: 使用I-XXX（如果存在）或-100 (推荐使用-100以避免复杂性)
- S-XXX的后续subword: 使用-100（单字符实体不应被拆分）
- E-XXX的后续subword: 使用-100
"""


def align_labels_with_tokens(labels, word_ids, label_list=None):
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id is None:
            # 特殊token（[CLS], [SEP], [PAD]等）
            new_labels.append(-100)
        elif word_id != current_word:
            # 新词的开始
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # 同一个词的后续subword
            # 如果提供了label_list，尝试进行BIOES标签转换
            if label_list is not None:
                label_name = label_list[labels[word_id]]
                # 对于B-XXX，尝试转换为I-XXX
                if label_name.startswith('B-'):
                    entity_type = label_name[2:]
                    inside_label = f'I-{entity_type}'
                    if inside_label in label_list:
                        new_labels.append(label_list.index(inside_label))
                    else:
                        new_labels.append(-100)
                else:
                    # 其他情况（I-XXX, S-XXX, E-XXX, O）都使用-100
                    new_labels.append(-100)
            else:
                # 如果没有label_list，统一使用-100
                new_labels.append(-100)

    return new_labels


def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,  # 重要：告诉tokenizer输入已经是分词后的
        padding=False,  # 在collator中进行padding
        max_length=512
    )

    all_labels = []
    for i, labels in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(i)
        # 将标签转换为ID
        label_ids = [label2id[label] for label in labels]
        # 对齐标签（传入标签列表以支持BIOES标签转换）
        label_list = list(label2id.keys())
        aligned_labels = align_labels_with_tokens(
            label_ids, word_ids, label_list)
        all_labels.append(aligned_labels)

    tokenized_inputs['labels'] = all_labels
    return tokenized_inputs


def compute_metrics(eval_preds, label_list):
    """
    计算NER评估指标
    使用seqeval库计算precision, recall, f1

    Args:
        eval_preds: (predictions, labels)元组
        label_list: 标签列表

    Returns:
        包含precision, recall, f1, accuracy的字典
    """
    predictions, labels = eval_preds

    # predictions是logits，需要取argmax
    predictions = np.argmax(predictions, axis=2)

    # 移除被忽略的标签（-100）并转换为标签名称
    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_prediction = []
        for pred, lab in zip(prediction, label):
            if lab != -100:
                true_label.append(label_list[lab])
                true_prediction.append(label_list[pred])
        true_labels.append(true_label)
        true_predictions.append(true_prediction)

    # 使用seqeval计算指标
    results = {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
    }

    return results


def export_to_onnx(model, tokenizer, output_path, example_text="北京是中国的首都"):
    """
    将模型导出为ONNX格式

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        output_path: ONNX模型输出路径
        example_text: 用于导出的示例文本
    """
    model.eval()

    # 准备示例输入
    inputs = tokenizer(
        example_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 动态轴配置
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }

    # 导出ONNX
    torch.onnx.export(
        model,
        (inputs['input_ids'].to(model.device),
         inputs['attention_mask'].to(model.device)),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )

    print(f"Model exported to {output_path}")

    # 验证ONNX模型
    try:
        import onnx
        import onnxruntime as ort

        # 检查ONNX模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

        # 测试ONNX推理
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        print(
            f"ONNX inference test passed! Output shape: {ort_outputs[0].shape}")

    except ImportError:
        print("Please install onnx and onnxruntime to validate the exported model:")
        print("pip install onnx onnxruntime")


def predict_ner(text, model, tokenizer, label_list, device='cpu'):
    """
    使用训练好的模型进行NER预测

    Args:
        text: 输入文本
        model: 训练好的模型
        tokenizer: 分词器
        label_list: 标签列表
        device: 设备

    Returns:
        包含token和对应标签的列表
    """
    model.eval()
    model.to(device)

    # 分词
    inputs = tokenizer(
        [item for item in text],
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    offset_mapping = inputs.pop('offset_mapping')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # 解码结果
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [label_list[pred]
                        for pred in predictions[0].cpu().numpy()]

    # 过滤特殊token
    results = []
    for token, label, offset in zip(tokens, predicted_labels, offset_mapping[0]):
        if token not in ['[CLS]', '[SEP]', '[PAD]'] and offset[0] != offset[1]:
            results.append({
                'token': token,
                'label': label,
                'start': offset[0].item(),
                'end': offset[1].item()
            })
    # 这里只打印了token级别的结果，实际应用中可能需要根据offset进行实体拼接, 还原文本
    return results


def main():

    # 下载模型命令 hf dowbload google-bert/bert-base-chinese --local-dir ./models/bert-base-chinese
    model_path = "./models/bert-base-chinese"
    data_dir = "./datasets/msra_ner"
    output_dir = "./output/ner_model"
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)

    # 1. 加载数据
    train_data = load_bio_format_data(os.path.join(data_dir, "train.txt"))
    val_data = load_bio_format_data(os.path.join(data_dir, "test.txt"))
    print(
        f"Loaded {len(train_data)} train examples, Loaded {len(val_data)} val examples")

    # 建立标签到id的映射
    all_data = train_data + val_data
    label_list = get_label_list(all_data)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    # 保存标签映射
    with open(os.path.join(output_dir, 'label2id.json'), 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # 转换为Dataset对象
    train_dataset = Dataset.from_dict({
        'tokens': [item['tokens'] for item in train_data],
        'ner_tags': [item['ner_tags'] for item in train_data]
    })

    val_datasets = Dataset.from_dict({
        'tokens': [item['tokens'] for item in val_data],
        'ner_tags': [item['ner_tags'] for item in val_data]
    })

    # 2. 进行分词和标签对齐
    tokenized_train = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_val = val_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=val_datasets.column_names
    )

    # 3. 加载预训练模型, 设置训练参数
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        warmup_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # 如果有GPU则使用混合精度
    )

    # 数据整理器, 用于动态padding
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 4. 创建Trainer并进行训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list)
    )
    trainer.train(resume_from_checkpoint=True)

    # 5. 预测部分示例

    test_texts = [
        "我来自北京大学，现在在上海工作。",
        "苹果公司的蒂姆·库克访问了中国。",
        "张三在清华大学学习计算机科学。"
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for text in test_texts:
        print(f"\nText: {text}")
        results = predict_ner(text, model, tokenizer, label_list, device)
        print("Predictions:")
        for result in results:
            print(f"  {result['token']: <10} -> {result['label']}")

    # 6. 导出为ONNX模型
    onnx_output_path = os.path.join(output_dir, "ner_model.onnx")
    export_to_onnx(model, tokenizer, onnx_output_path,
                   example_text=test_texts[0])

    print(f"Model saved to: {output_dir}")
    print(f"ONNX model saved to: {onnx_output_path}")


if __name__ == "__main__":
    main()
