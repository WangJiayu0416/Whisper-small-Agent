"""
阶段5：推理速度Benchmark
- 对比原版Whisper vs faster-whisper的推理速度
- 计算RTF（Real Time Factor）
- RTF < 1 表示能实时处理

面试必问：「你的模型推理延迟多少？能上线吗？」
这个脚本就是你的答案
"""

import yaml
import time
import torch
import numpy as np
from pathlib import Path


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def benchmark_original_whisper(model_path, test_audios, device="cuda"):
    """
    测试原版 HuggingFace Whisper 的推理速度
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("加载 HuggingFace Whisper...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    model.eval()

    latencies = []
    audio_durations = []

    with torch.no_grad():
        for audio, sr in test_audios:
            audio_duration = len(audio) / sr
            audio_durations.append(audio_duration)

            # 预处理
            input_features = processor(
                audio, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(device)

            # 推理计时
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            predicted_ids = model.generate(input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latency = end - start
            latencies.append(latency)

    return latencies, audio_durations


def benchmark_faster_whisper(model_path, test_audios):
    """
    测试 faster-whisper (CTranslate2) 的推理速度
    """
    from faster_whisper import WhisperModel

    print("加载 faster-whisper...")
    model = WhisperModel(model_path, device="cuda", compute_type="int8")

    latencies = []
    audio_durations = []

    for audio, sr in test_audios:
        audio_duration = len(audio) / sr
        audio_durations.append(audio_duration)

        start = time.perf_counter()

        segments, _ = model.transcribe(audio, language="zh")
        # 必须消费generator才能完成推理
        text = " ".join([seg.text for seg in segments])

        end = time.perf_counter()

        latency = end - start
        latencies.append(latency)

    return latencies, audio_durations


def run_benchmark(config):
    """
    运行完整benchmark

    输出：
    ┌──────────────────┬──────────┬──────────┬──────────┐
    │ 方案              │ 平均延迟  │ RTF      │ 能实时？  │
    ├──────────────────┼──────────┼──────────┼──────────┤
    │ HuggingFace FP32 │ 3.12s    │ 0.31     │ ✓        │
    │ faster-whisper   │ 0.89s    │ 0.09     │ ✓✓✓      │
    │ 加速比            │ 3.5x     │          │          │
    └──────────────────┴──────────┴──────────┴──────────┘
    """

    print("=" * 60)
    print("阶段5：推理速度 Benchmark")
    print("=" * 60)

    # ---- 准备测试数据 ----
    print("\n准备测试音频...")
    # 生成不同长度的合成测试音频（实际使用时替换为真实音频）
    sr = 16000
    test_durations = [3, 5, 10, 15, 20]  # 秒
    test_audios = []
    for dur in test_durations:
        audio = np.random.randn(sr * dur).astype(np.float32) * 0.1
        test_audios.append((audio, sr))
    print(f"  测试音频: {len(test_audios)} 条, 时长: {test_durations}")

    n_warmup = 2
    n_repeat = 3

    results = {}

    # ---- 1. 测试原版Whisper ----
    merged_path = "./outputs/merged_model"
    if Path(merged_path).exists():
        print(f"\n{'='*40}")
        print("测试 HuggingFace Whisper (FP32)")
        print(f"{'='*40}")

        # Warmup
        print(f"  Warmup ({n_warmup} runs)...")
        benchmark_original_whisper(merged_path, test_audios[:n_warmup])

        # Actual benchmark
        all_latencies = []
        all_durations = []
        for _ in range(n_repeat):
            lat, dur = benchmark_original_whisper(merged_path, test_audios)
            all_latencies.extend(lat)
            all_durations.extend(dur)

        avg_latency = np.mean(all_latencies)
        avg_rtf = np.mean([l / d for l, d in zip(all_latencies, all_durations)])
        results["hf_fp32"] = {"latency": avg_latency, "rtf": avg_rtf}
        print(f"  平均延迟: {avg_latency:.3f}s")
        print(f"  平均RTF: {avg_rtf:.4f}")

    # ---- 2. 测试faster-whisper ----
    ct2_path = "./outputs/faster_whisper_model"
    if Path(ct2_path).exists():
        print(f"\n{'='*40}")
        print("测试 faster-whisper (INT8)")
        print(f"{'='*40}")

        print(f"  Warmup ({n_warmup} runs)...")
        benchmark_faster_whisper(ct2_path, test_audios[:n_warmup])

        all_latencies = []
        all_durations = []
        for _ in range(n_repeat):
            lat, dur = benchmark_faster_whisper(ct2_path, test_audios)
            all_latencies.extend(lat)
            all_durations.extend(dur)

        avg_latency = np.mean(all_latencies)
        avg_rtf = np.mean([l / d for l, d in zip(all_latencies, all_durations)])
        results["faster_whisper_int8"] = {"latency": avg_latency, "rtf": avg_rtf}
        print(f"  平均延迟: {avg_latency:.3f}s")
        print(f"  平均RTF: {avg_rtf:.4f}")

    # ---- 3. 汇总对比 ----
    if len(results) >= 2:
        speedup = results["hf_fp32"]["latency"] / results["faster_whisper_int8"]["latency"]
        print(f"\n{'='*50}")
        print("Benchmark 结果汇总")
        print(f"{'='*50}")
        print(f"{'方案':<25} {'延迟(s)':>10} {'RTF':>10}")
        print("-" * 47)
        print(f"{'HuggingFace FP32':<25} {results['hf_fp32']['latency']:>9.3f}s {results['hf_fp32']['rtf']:>9.4f}")
        print(f"{'faster-whisper INT8':<25} {results['faster_whisper_int8']['latency']:>9.3f}s {results['faster_whisper_int8']['rtf']:>9.4f}")
        print(f"\n加速比: {speedup:.1f}x")

    # 保存结果
    import json
    with open("./outputs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    config = load_config()
    run_benchmark(config)
