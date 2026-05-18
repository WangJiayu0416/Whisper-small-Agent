"""
阶段2：噪声增强 + 课程学习数据分级
- 对训练数据添加不同等级的噪声
- 按SNR分级标注，供课程学习使用

这是你的创新点之一：不是简单加噪，而是按难度分级
"""

import numpy as np
import yaml
import gc
from datasets import load_from_disk,Dataset, concatenate_datasets
import soundfile as sf
import os
import copy
import shutil

def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_gaussian_noise(audio, snr_db):
    """添加高斯白噪声，指定信噪比"""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def add_pink_noise(audio, snr_db):
    """添加粉红噪声（低频更强，更接近真实环境）"""
    n_samples = len(audio)
    # 生成粉红噪声：频率越高幅度越小
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # 避免除以0
    pink_spectrum = 1 / np.sqrt(freqs)
    white_noise = np.random.randn(n_samples)
    pink_noise = np.fft.irfft(np.fft.rfft(white_noise) * pink_spectrum, n=n_samples)

    # 调整到目标SNR
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    pink_noise = pink_noise * np.sqrt(noise_power / (np.mean(pink_noise ** 2) + 1e-10))
    return audio + pink_noise


# 噪声生成函数映射
NOISE_FUNCTIONS = {
    "gaussian": add_gaussian_noise,
    "pink": add_pink_noise,
}


def augment_with_noise(example, snr_db, noise_type="gaussian"):
    """对单条数据添加噪声并标注SNR等级"""
    try:
        audio = example["audio"]["array"].astype(np.float32)

        # 检查音频是否有效
        if len(audio) == 0 or np.all(audio == 0):
            print(f"  跳过静音/空音频")
            return None

        noise_fn = NOISE_FUNCTIONS.get(noise_type, add_gaussian_noise)
        noisy_audio = noise_fn(audio, snr_db)

        # 防止clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val

        example["audio"]["array"] = noisy_audio
        example["snr_db"] = snr_db          # 标注SNR，课程学习要用
        example["noise_type"] = noise_type  # 标注噪声类型
        return example

    except Exception as e:
        print(f"  加噪失败，跳过：{e}")
        return None


def create_noisy_dataset(config):
    """
    生成带噪声标注的增强数据集

    关键设计：每条数据都标注了snr_db，
    课程学习训练时按snr_db筛选不同难度的数据
    """

    print("=" * 60)
    print("噪声增强 + 课程学习数据分级")
    print("=" * 60)

    dataset = load_from_disk("/root/autodl-tmp/data/processed_dataset")
    train_data = dataset["train"]

    snr_levels = config["noise_augment"]["snr_levels"]
    noise_types = config["noise_augment"]["noise_types"]

    print(f"\nSNR等级: {snr_levels}")
    print(f"噪声类型: {noise_types}")

    # 原始干净数据标注为 snr=999（代表无噪声）
    all_examples = []

    print("\n[1] 保留原始干净数据...")
    for i, example in enumerate(train_data):
        example["snr_db"] = 999
        example["noise_type"] = "clean"
        all_examples.append(example)

    # 对每个SNR等级 × 每种噪声类型生成增强数据
    print("[2] 生成带噪数据...")
    augment_ratio = config["noise_augment"]["augment_ratio"]
    n_augment = int(len(train_data) * augment_ratio)

    for snr in snr_levels:
        for noise_type in noise_types:
            if noise_type == "babble":
                noise_type_actual = "gaussian"  # 简化处理
            else:
                noise_type_actual = noise_type

            # 随机采样一部分数据做增强
            indices = np.random.choice(len(train_data), size=n_augment, replace=False)
            for idx in indices:
                example = copy.deepcopy(train_data[int(idx)])
                result = augment_with_noise(example, snr, noise_type_actual)
                if result is not None:          # 过滤掉加噪失败的数据
                    all_examples.append(result)

            print(f"  SNR={snr}dB, 噪声={noise_type}: +{n_augment} 条")

    print(f"\n总数据量: {len(all_examples)} 条")
    print(f"  其中干净数据: {len(train_data)} 条")
    print(f"  其中增强数据: {len(all_examples) - len(train_data)} 条")

    # 统计各SNR等级的数据分布
    snr_counts = {}
    for ex in all_examples:
        snr = ex["snr_db"]
        snr_counts[snr] = snr_counts.get(snr, 0) + 1
    print("\nSNR分布:")
    for snr, count in sorted(snr_counts.items()):
        label = "clean" if snr == 999 else f"{snr}dB"
        print(f"  {label}: {count} 条")

    # 保存前释放list内存，避免同时占用双份内存
    print("\n开始保存数据集...")
    save_path = "/root/autodl-tmp/data/augmented_train_dataset_parts"
    os.makedirs(save_path, exist_ok=True)

    batch_size = 2000
    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i:i+batch_size]
        batch_ds = Dataset.from_list(batch)
        part_path = f"{save_path}/part_{i//batch_size}"
        batch_ds.save_to_disk(part_path)
        del batch, batch_ds
        gc.collect()
        print(f"  已保存 {min(i+batch_size, len(all_examples))}/{len(all_examples)} 条")

    del all_examples
    gc.collect()
    
# 合并所有分片
    print("\n合并分片...")
    parts = []
    for i in range(0, 22694, batch_size):
        part_path = f"{save_path}/part_{i//batch_size}"
        if os.path.exists(part_path):
            parts.append(load_from_disk(part_path))

    combined = concatenate_datasets(parts)
    combined.save_to_disk("/root/autodl-tmp/data/augmented_train_dataset")
    print(f"\n合并完成，共 {len(combined)} 条")
    print("增强数据集已保存到 /root/autodl-tmp/data/augmented_train_dataset")

    
    return augmented_dataset


if __name__ == "__main__":
    config = load_config()
    create_noisy_dataset(config)