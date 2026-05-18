
"""
阶段2：数据准备
- 下载Common Voice中文数据集
- 过滤异常音频（过长/过短/静音）
- 统一采样率为16kHz
"""

import yaml
from datasets import load_dataset, Audio
import numpy as np


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_by_duration(example, min_sec, max_sec):
    """过滤过长或过短的音频"""
    duration = len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    return min_sec <= duration <= max_sec


def remove_special_chars(example):
    """清洗文本：去掉标点等干扰"""
    import re
    # 保留中文字符、英文字母和数字
    text = example["sentence"]
    text = re.sub(r"[^\u4e00-\u9fff\w\s]", "", text)
    text = text.strip()
    example["sentence"] = text
    return example


def normalize_columns(dataset, text_column):
    """把不同数据集的转写字段统一重命名为 sentence，便于下游处理"""
    if text_column == "sentence":
        return dataset
    for split in dataset.keys():
        if text_column in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column(text_column, "sentence")
    return dataset


def prepare_dataset(config):
    """主函数：下载 + 清洗 + 保存"""

    print("=" * 60)
    print("阶段2：数据准备")
    print("=" * 60)

    # ---- 1. 下载数据集 ----
    print("\n[1/4] 下载数据集...")
    dataset = load_dataset(
        config["data"]["dataset_name"],
        config["data"]["language"],
    )

    # 统一字段名到 sentence，兼容 FLEURS (transcription) 等其他数据集
    text_column = config["data"].get("text_column", "sentence")
    dataset = normalize_columns(dataset, text_column)

    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  验证集: {len(dataset['validation'])} 条")
    print(f"  测试集: {len(dataset['test'])} 条")

    # ---- 2. 统一采样率 ----
    print("\n[2/4] 统一采样率为 16kHz...")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=config["data"]["sampling_rate"])
    )

    # ---- 3. 过滤异常音频 ----
    print("\n[3/4] 过滤异常音频...")
    min_sec = config["data"]["min_duration_sec"]
    max_sec = config["data"]["max_duration_sec"]

    for split in ["train", "validation", "test"]:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda x: filter_by_duration(x, min_sec, max_sec)
        )
        after = len(dataset[split])
        print(f"  {split}: {before} → {after} （过滤了 {before - after} 条）")

    # ---- 4. 清洗文本 ----
    print("\n[4/4] 清洗文本...")
    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(remove_special_chars)
        # 过滤空文本
        dataset[split] = dataset[split].filter(lambda x: len(x["sentence"]) > 0)

    # ---- 保存 ----
    save_path = "/root/autodl-tmp/data/processed_dataset"
    print(f"\n保存到 {save_path}...")
    dataset.save_to_disk(save_path)

    print("\n数据准备完成！")
    print(f"  最终训练集: {len(dataset['train'])} 条")
    print(f"  最终验证集: {len(dataset['validation'])} 条")
    print(f"  最终测试集: {len(dataset['test'])} 条")

    return dataset


if __name__ == "__main__":
    config = load_config()
    prepare_dataset(config)
