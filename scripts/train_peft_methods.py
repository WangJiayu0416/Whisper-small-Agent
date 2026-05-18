"""
阶段4：PEFT 方法对比实验
- 在相同配置下分别训练 LoRA / DoRA / AdaLoRA
- 对比 CER、可训练参数、训练时间、显存

LoRA:    W' = W + B·A，固定 rank
DoRA:    把权重分解为方向和幅度，只对方向做 LoRA（ICML 2024 Oral）
AdaLoRA: 训练时根据各层重要性动态调整 rank 预算，重要的层多给 rank
"""

import gc
import json
import time
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, AdaLoraConfig, get_peft_model, TaskType

# 复用 LoRA 脚本中的工具类
from train_lora import DataCollatorSpeechSeq2SeqWithPadding, compute_wer


def load_config(path="/root/whisper-finetune/configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_peft_model(model, method, config, total_steps=None):
    """根据 method 添加对应的 PEFT 适配器"""

    if method == "lora":
        peft_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias="none",
        )

    elif method == "dora":
        # DoRA 只需在 LoRA 基础上加 use_dora=True
        peft_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias="none",
            use_dora=True,
        )

    elif method == "adalora":
        # AdaLoRA 的三阶段调度
        # tinit:  前 tinit 步用初始 rank 训练（预热）
        # tfinal: 最后 tfinal 步只 finetune 不再调整 rank
        # 中间阶段动态减少 rank 到 target_r
        peft_config = AdaLoraConfig(
            init_r=12,                   # 初始 rank
            target_r=8,                  # 目标 rank（跟我们 LoRA r=8 对齐）
            tinit=100,
            tfinal=200,
            deltaT=10,
            beta1=0.85,
            beta2=0.85,
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            total_step=total_steps,
        )

    else:
        raise ValueError(f"未知方法: {method}")

    return get_peft_model(model, peft_config)


def train_one_method(method, config):
    """训练单个 PEFT 方法"""

    print("=" * 60)
    print(f"  PEFT 方法: {method.upper()}")
    print("=" * 60)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. 加载基座模型 ----
    model_name = config["model"]["name"]
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=config["model"]["language"],
        task=config["model"]["task"],
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config["model"]["language"],
        task=config["model"]["task"],
    )
    model.config.suppress_tokens = []

    # ---- 2. 加载数据 ----
    dataset = load_from_disk("/root/autodl-tmp/data/whisper_dataset")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # AdaLoRA 需要知道总训练步数
    # 总步数 ≈ epochs × ceil(train_size / (batch_size × grad_accum))
    train_size = len(dataset["train"])
    bs = config["training"]["batch_size"]
    ga = config["training"]["gradient_accumulation_steps"]
    epochs = config["training"]["num_epochs"]
    total_steps = (train_size // (bs * ga)) * epochs

    # ---- 3. 添加 PEFT ----
    model = build_peft_model(model, method, config, total_steps=total_steps)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # ---- 4. 训练参数 ----
    output_dir = f"/root/whisper-finetune/outputs/peft_{method}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        fp16=config["training"]["fp16"],
        eval_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_wer(pred, processor),
        processing_class=processor.feature_extractor,
    )

    print("\n开始训练...")
    trainer.train()

    # ---- 5. 评估 ----
    test_results = trainer.evaluate(dataset["test"])
    print(f"\n测试集 CER: {test_results['eval_wer']*100:.2f}%")

    elapsed_min = (time.time() - start_time) / 60
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0
    )

    print(f"训练时间: {elapsed_min:.1f} min")
    print(f"GPU 峰值显存: {gpu_peak_mb:.0f} MB")

    # ---- 6. 保存 ----
    trainer.save_model(f"{output_dir}/best_model")
    processor.save_pretrained(f"{output_dir}/best_model")

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "method": method,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / total,
            "test_wer": test_results["eval_wer"],
            "training_time_min": elapsed_min,
            "gpu_peak_mb": gpu_peak_mb,
        }, f, indent=2)

    # ---- 7. 释放显存 ----
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": method,
        "cer": test_results["eval_wer"],
        "trainable": trainable,
        "trainable_ratio": trainable / total,
        "time_min": elapsed_min,
        "gpu_mb": gpu_peak_mb,
    }


def run_all(config, methods=None):
    """依次跑指定的 PEFT 方法"""
    if methods is None:
        methods = ["dora", "adalora"]   # lora 已跑过，默认跑剩下两个

    all_results = []
    for m in methods:
        result = run_with_oom_retry(m, config)
        all_results.append(result)

    # ---- 汇总输出 ----
    print("\n" + "=" * 70)
    print("PEFT 方法对比结果")
    print("=" * 70)
    print(f"{'方法':<12} {'CER':>8} {'可训练参数':>14} {'占比':>8} {'时间':>10} {'显存':>10}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['method'].upper():<12} "
            f"{r['cer']*100:>7.2f}% "
            f"{r['trainable']:>13,} "
            f"{r['trainable_ratio']*100:>7.2f}% "
            f"{r['time_min']:>9.1f}m "
            f"{r['gpu_mb']:>9.0f}M"
        )

    with open("/root/whisper-finetune/outputs/peft_methods_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\n结果保存到 outputs/peft_methods_results.json")

    return all_results


def run_with_oom_retry(method, config):
    """跑一个方法，OOM 时清显存重试一次"""
    try:
        return train_one_method(method, config)
    except torch.cuda.OutOfMemoryError:
        print(f"\n[警告] {method} 训练 OOM，清空显存后重试...")
        gc.collect()
        torch.cuda.empty_cache()
        return train_one_method(method, config)


if __name__ == "__main__":
    config = load_config()
    # 默认只跑 dora 和 adalora，因为 lora 已经在 outputs/lora_rank8 跑过
    run_all(config, methods=["adalora"])
