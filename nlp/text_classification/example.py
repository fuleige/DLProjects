from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

import numpy as np

"""
数据集格式处理:
    对于文本分类任务, 将其整理成jsonl格式, 这种格式每一行都是一个标准的json字符串, 每个json必须包含两个字段: text和label
        label是类别编号, 从0开始; text是文本内容
    一般的jsonl格式内容如下:
        {"text": "I loved this movie!", "label": 1}
        {"text": "This film was terrible.", "label": 0}
    
然后是切分数据集,这里可以简单分为训练集和验证集
    每个类目的数据打乱顺序后, 按照8:2的比例划分为训练集和验证集
    训练集和验证集分别保存为train.jsonl和val.jsonl文件
"""


def main():

    # 需要预下载
    # 从huggingface模型库到 models 目录下的示例命令如下
    # hf download distilbert/distilbert-base-uncased --local-dir models/distilbert-base-uncased
    model_path = "./models/distilbert-base-uncased"
    num_labels = 2

    # 1. 加载数据集
    datasets_files = {"train": "datasets/train.jsonl",
                      "val": "datasets/val.jsonl"}
    datasets = load_dataset("json", data_files=datasets_files)

    # 2. 加载分词器,并且对数据集进行分词处理
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # 数据处理器, 由于文本长度不一, 需要动态填充对齐形成一批次输入
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # # 3. 加载模型, 定义训练参数
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="my_model",  # 保存训练中间结果和最终模型的目录
        learning_rate=2e-5,    # 学习率
        per_device_train_batch_size=16,  # 训练批次大小 (根据显存调整)
        per_device_eval_batch_size=16,  # 验证批次大小
        num_train_epochs=2,  # 训练轮数, 这个轮数可以根据数据集大小和效果调整
        weight_decay=0.01,     # 权重衰减系数
        eval_strategy="epoch",  # 评估策略, 每个epoch结束后评估一次
        save_strategy="epoch",  # 保存策略, 每个epoch结束后保存一次
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
    )

    # 4. 定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    # # 5. 创建Trainer并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # resume_from_checkpoint: 是否从上次中断的地方继续训练
    trainer.train(resume_from_checkpoint=True)

    # # 6. 测试单个样本
    labels = ["差评", "好评"]
    for test_text in ["味道比较一般", "非常好吃，下次还会光顾", "配送速度好快,服务可以的", "食物都凉了"]:
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
        print(f"Text: {test_text} => Predicted label: {labels[predictions[0]]}")
        
    # 可以借助 optimum 库将模型转换为 ONNX 格式以方便部署


if __name__ == "__main__":
    main()
