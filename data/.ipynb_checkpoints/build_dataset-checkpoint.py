"""
阶段2：构建HuggingFace Dataset
- 加载 processed_dataset（纯净音频）
- 转换为 Whisper 所需格式：input_features + labels
- 保存为 whisper_dataset，供 train_lora.py 使用
"""

import yaml
from datasets import load_from_disk, DatasetDict
from transformers import WhisperProcessor

MAX_AUDIO_SAMPLES = 16000 * 30  # Whisper 最大支持30s
MAX_LABEL_TOKENS = 448          # Whisper decoder 最大token数


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(config):
    print("=" * 60)
    print("构建 Whisper 训练数据集（纯净音频）")
    print("=" * 60)

    model_name = config["model"]["name"]
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=config["model"]["language"],
        task=config["model"]["task"],
    )

    # ---- 1. 加载纯净数据集 ----
    print("\n[1] 加载纯净数据集...")
    dataset = load_from_disk("/root/autodl-tmp/data/processed_dataset")
    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  验证集: {len(dataset['validation'])} 条")
    print(f"  测试集: {len(dataset['test'])} 条")

    # ---- 2. 特征提取 ----
    def prepare_features(example):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        # 超过30s截断（完整音频可能超限）
        if len(audio_array) > MAX_AUDIO_SAMPLES:
            audio_array = audio_array[:MAX_AUDIO_SAMPLES]

        # 音频 → log-mel spectrogram，shape: (80, 3000)
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
        ).input_features[0]

        # 文本 → token ids
        labels = processor.tokenizer(
            example["sentence"],
            truncation=True,
            max_length=MAX_LABEL_TOKENS,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    # ---- 3. 处理各split ----
    print("\n[2] 处理数据中（这步比较慢，请耐心等待）...")
    final_dataset = DatasetDict()

    for split in ["train", "validation", "test"]:
        print(f"\n  处理 {split}...")
        final_dataset[split] = dataset[split].map(
            prepare_features,
            remove_columns=dataset[split].column_names,
        )
        print(f"  完成，{len(final_dataset[split])} 条")

    # ---- 4. 保存 ----
    save_path = "/root/autodl-tmp/data/whisper_dataset"
    print(f"\n[3] 保存到 {save_path}...")
    final_dataset.save_to_disk(save_path)
    print("\n完成！")

    return final_dataset


if __name__ == "__main__":
    config = load_config()
    build_dataset(config)

