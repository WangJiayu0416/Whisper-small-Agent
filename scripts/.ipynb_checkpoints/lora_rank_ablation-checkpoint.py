"""
阶段4：LoRA Rank 消融实验
- 遍历 rank = [2, 4, 8, 16, 32]，每个rank训练一个模型
- 记录 WER、可训练参数量、训练时间、显存峰值
- 输出对比表 + Pareto曲线数据

这是你的创新点之一：不是拍脑袋选rank，而是有数据支撑
"""

import yaml
import json
import time
import torch
from train_lora import train_lora


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_rank_ablation(config):
    """
    依次训练不同rank的LoRA，收集所有结果

    最终输出类似：
    ┌───────┬────────┬──────────────┬───────────┐
    │ Rank  │ WER    │ 可训练参数    │ 训练时间   │
    ├───────┼────────┼──────────────┼───────────┤
    │ 2     │ 17.1%  │ 590K (0.24%) │ 45min     │
    │ 4     │ 15.9%  │ 1.2M (0.48%) │ 48min     │
    │ 8     │ 15.4%  │ 2.4M (0.95%) │ 52min     │
    │ 16    │ 15.2%  │ 4.7M (1.90%) │ 58min     │
    │ 32    │ 15.1%  │ 9.4M (3.80%) │ 65min     │
    └───────┴────────┴──────────────┴───────────┘
    结论：rank=8是最优平衡点（WER拐点）
    """

    print("=" * 60)
    print("LoRA Rank 消融实验")
    print("=" * 60)

    ranks = config["lora"]["ablation_ranks"]
    all_results = []

    for rank in ranks:
        print(f"\n{'='*40}")
        print(f"  训练 rank={rank}")
        print(f"{'='*40}")

        start_time = time.time()

        # 记录GPU显存峰值
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 训练
        test_results = train_lora(config, lora_rank=rank)
    

        elapsed = time.time() - start_time
        gpu_peak_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )

        gc.collect()
        torch.cuda.empty_cache()

        # 读取该rank的详细结果
        with open(f"./outputs/lora_rank{rank}/results.json", "r") as f:
            detailed = json.load(f)

        result = {
            "rank": rank,
            "wer": detailed["test_wer"],
            "trainable_params": detailed["trainable_params"],
            "total_params": detailed["total_params"],
            "trainable_ratio": detailed["trainable_params"] / detailed["total_params"],
            "training_time_min": elapsed / 60,
            "gpu_peak_mb": gpu_peak_mb,
        }
        all_results.append(result)

        print(f"\n  rank={rank} 完成:")
        print(f"    WER: {result['wer']*100:.2f}%")
        print(f"    可训练参数: {result['trainable_params']:,} ({result['trainable_ratio']*100:.2f}%)")
        print(f"    训练时间: {result['training_time_min']:.1f} min")
        print(f"    GPU峰值显存: {result['gpu_peak_mb']:.0f} MB")

    # ---- 输出汇总表 ----
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(f"{'Rank':<8} {'WER':>8} {'参数量':>14} {'占比':>8} {'时间(min)':>10} {'显存(MB)':>10}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['rank']:<8} "
            f"{r['wer']*100:>7.2f}% "
            f"{r['trainable_params']:>13,} "
            f"{r['trainable_ratio']*100:>7.2f}% "
            f"{r['training_time_min']:>9.1f} "
            f"{r['gpu_peak_mb']:>9.0f}"
        )

    # 保存结果
    with open("./outputs/rank_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n结果已保存到 ./outputs/rank_ablation_results.json")

    # ---- 找最优rank ----
    # 简单策略：找WER下降的拐点
    for i in range(1, len(all_results)):
        prev_wer = all_results[i - 1]["wer"]
        curr_wer = all_results[i]["wer"]
        improvement = (prev_wer - curr_wer) / prev_wer * 100
        if improvement < 1.0:  # WER提升不到1%，认为到了拐点
            best_rank = all_results[i - 1]["rank"]
            print(f"\n建议最优 rank = {best_rank}")
            print(f"  理由：rank从{best_rank}增大到{all_results[i]['rank']}时，")
            print(f"  WER仅提升{improvement:.2f}%，但参数量增加了一倍")
            break

    return all_results


if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)
    config = load_config()
    run_rank_ablation(config)
