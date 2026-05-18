"""
阶段4：全参数微调（对比实验用）
- 冻结encoder前N层，其余全部训练
- 与LoRA对比：WER差多少？显存差多少？训练时间差多少？
"""

import yaml
import torch
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import gc
import time

# 复用LoRA脚本中的工具类
from train_lora import DataCollatorSpeechSeq2SeqWithPadding, compute_wer


def load_config(path="/root/whisper-finetune/configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_full_finetune(config):
    """
    全参数微调（冻结encoder前几层）

    跟LoRA的区别：
    - LoRA: 冻结全部原始参数，只训练旁路小矩阵
    - 这里: 冻结前4层encoder，其余所有参数都训练
    """

    print("=" * 60)
    print("阶段4：全参数微调（冻结前N层encoder）")
    print("=" * 60)
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. 加载模型 ----
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

    # ---- 2. 冻结encoder前N层 ----
    n_freeze = config["freeze"]["encoder_layers_to_freeze"]
    print(f"\n冻结 encoder 前 {n_freeze} 层...")

    # 冻结embedding层
    for param in model.model.encoder.embed_positions.parameters():
        param.requires_grad = False
    for param in model.model.encoder.conv1.parameters():
        param.requires_grad = False
    for param in model.model.encoder.conv2.parameters():
        param.requires_grad = False

    # 冻结前N层transformer
    for i, layer in enumerate(model.model.encoder.layers):
        if i < n_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ---- 3. 训练 ----
    dataset = load_from_disk("/root/autodl-tmp/data/whisper_dataset")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    output_dir = "/root/whisper-finetune/outputs/full_finetune"
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

    # ---- 4. 评估 ----
    trainer.save_model(f"{output_dir}/best_model")
    processor.save_pretrained(f"{output_dir}/best_model")

    test_results = trainer.evaluate(dataset["test"])
    print(f"\n测试集 WER: {test_results['eval_wer']*100:.2f}%")

    elapsed_min = (time.time() - start_time) / 60
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0
    )
    print(f"训练时间: {elapsed_min:.1f} min")
    print(f"GPU 峰值显存: {gpu_peak_mb:.0f} MB")

    import json
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "method": "full_finetune",
            "frozen_layers": n_freeze,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / total,
            "test_wer": test_results["eval_wer"],
            "training_time_min": elapsed_min,
            "gpu_peak_mb": gpu_peak_mb,
        }, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return test_results

    return test_results


if __name__ == "__main__":
    config = load_config()
    train_full_finetune(config)
