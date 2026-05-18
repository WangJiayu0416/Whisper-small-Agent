"""
阶段3：跑原版Whisper基线
- 不做任何fine-tune，直接用原版Whisper在测试集上跑
- 记录WER作为baseline，后面所有实验都跟这个比
- 分别在干净测试集和不同SNR噪声测试集上评估

没有baseline的实验等于没有实验！
"""

import yaml
import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_on_dataset(model, processor, dataset, device):
    wer_metric = evaluate.load("wer")
    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        for example in dataset:
            input_features = torch.tensor(
                example["input_features"]
            ).unsqueeze(0).to(device)

            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="zh", task="transcribe")
            )
            pred_text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].replace(" ", "")

            ref_text = processor.tokenizer.decode(
                example["labels"], skip_special_tokens=True
            ).replace(" ", "")

            predictions.append(" ".join(list(pred_text)))
            references.append(" ".join(list(ref_text)))

    cer = wer_metric.compute(predictions=predictions, references=references)
    return cer


def run_baseline(config):
    """
    运行基线实验，输出一张对比表

    这张表是你整个项目的起点：
    ┌──────────────┬────────┐
    │ 测试条件      │ WER    │
    ├──────────────┼────────┤
    │ 干净测试集    │ xx.x%  │
    │ SNR=20dB     │ xx.x%  │
    │ SNR=10dB     │ xx.x%  │
    │ SNR=5dB      │ xx.x%  │
    └──────────────┴────────┘
    """

    print("=" * 60)
    print("阶段3：Whisper 原版基线评估")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # 加载模型和数据
    model_name = config["model"]["name"]
    print(f"加载模型: {model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)

    dataset = load_from_disk("/root/autodl-tmp/data/whisper_dataset")
    test_data = dataset["test"]

    # 只取前200条做快速评估（完整评估太慢）
    n_eval = min(200, len(test_data))
    test_subset = test_data.select(range(n_eval))
    print(f"评估样本数: {n_eval}")

    # ---- 1. 干净测试集 ----
    print("\n[1/4] 评估干净测试集...")
    wer_clean = evaluate_on_dataset(model, processor, test_subset, device)

    # ---- 2-4. 不同SNR的噪声测试集 ----
    snr_levels = [20, 10, 5]
    results = {"clean": wer_clean}

    for snr in snr_levels:
        print(f"\n[评估 SNR={snr}dB 测试集...]")
        # TODO: 在实际实现中，需要对测试集音频重新加噪
        # 这里展示的是评估框架，你需要在原始音频上加噪后重新提取特征
        # noisy_test = add_noise_to_dataset(test_subset, snr)
        # wer = evaluate_on_dataset(model, processor, noisy_test, device)
        # results[f"SNR={snr}dB"] = wer
        print(f"  （需要实现：对原始音频加噪后重新提取特征）")

    # ---- 输出结果 ----
    print("\n" + "=" * 40)
    print("基线结果")
    print("=" * 40)
    print(f"{'测试条件':<20} {'CER':>10}")
    print("-" * 32)
    for condition, wer in results.items():
        print(f"{condition:<20} {wer*100:>9.2f}%")
    print("=" * 40)

    # 保存结果
    import json
    results_serializable = {k: float(v) for k, v in results.items()}
    with open("./outputs/baseline_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print("\n结果已保存到 ./outputs/baseline_results.json")

    return results


if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)
    config = load_config()
    run_baseline(config)
