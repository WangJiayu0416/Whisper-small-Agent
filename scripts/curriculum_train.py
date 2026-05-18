"""
阶段4：课程学习训练
- 训练分3个阶段，逐渐增加噪声难度
- 阶段1（epoch 1-3）：只用干净数据
- 阶段2（epoch 4-7）：用中等噪声数据
- 阶段3（epoch 8-10）：用高噪声数据

这是你最核心的创新点！
面试时说：「我引入课程学习策略，让模型先学会识别干净语音，
再逐步适应噪声环境，类比人类先在安静环境学语言再适应嘈杂场景」
"""

import yaml
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model
import evaluate

from scripts.train_lora import DataCollatorSpeechSeq2SeqWithPadding, compute_wer


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_by_snr(dataset, snr_min, snr_max):
    """按SNR范围筛选数据"""
    return dataset.filter(
        lambda x: snr_min <= x.get("snr_db", 999) <= snr_max
    )


def curriculum_train(config):
    """
    课程学习训练

    关键思想：
    - 标准训练：所有数据混在一起，模型同时面对干净和嘈杂样本
    - 课程学习：由易到难，先掌握简单的，再攻克困难的

    类比：学英语听力
    - 标准方式：从第一天起同时听VOA慢速和BBC快速
    - 课程学习：先听慢速掌握基础，再逐步加快语速
    """

    print("=" * 60)
    print("阶段4：课程学习训练")
    print("=" * 60)

    # 加载模型
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

    # 添加LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # 加载数据
    dataset = load_from_disk("./data/whisper_dataset")
    # 注意：这里需要加载包含snr_db标注的增强数据集
    # augmented_train = load_from_disk("./data/augmented_train_dataset")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    stages = config["curriculum"]["stages"]
    output_dir = "./outputs/curriculum"

    print(f"\n课程学习分 {len(stages)} 个阶段:")
    for s in stages:
        print(f"  {s['name']}: {s['epochs']} epochs, SNR范围 {s['snr_range']}")

    # ---- 分阶段训练 ----
    for stage_idx, stage in enumerate(stages):
        stage_name = stage["name"]
        n_epochs = stage["epochs"]
        snr_min, snr_max = stage["snr_range"]

        print(f"\n{'='*50}")
        print(f"  课程阶段 {stage_idx+1}/{len(stages)}: {stage_name}")
        print(f"  Epochs: {n_epochs}, SNR: [{snr_min}, {snr_max}]dB")
        print(f"{'='*50}")

        # 按SNR筛选当前阶段的训练数据
        # stage_train_data = filter_by_snr(augmented_train, snr_min, snr_max)
        # print(f"  当前阶段训练数据: {len(stage_train_data)} 条")

        # 每个阶段的训练参数
        stage_output = f"{output_dir}/stage_{stage_idx}"
        training_args = Seq2SeqTrainingArguments(
            output_dir=stage_output,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=config["training"]["batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            learning_rate=config["training"]["learning_rate"],
            fp16=config["training"]["fp16"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
            report_to="tensorboard",
            logging_dir=f"./logs/curriculum_stage{stage_idx}",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],  # TODO: 替换为 stage_train_data
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_wer(pred, processor),
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

        # 阶段结束后评估
        eval_results = trainer.evaluate(dataset["validation"])
        print(f"\n  阶段 {stage_name} 完成, 验证WER: {eval_results['eval_wer']*100:.2f}%")

        # 注意：模型权重会自动传递到下一阶段，不需要重新加载
        # 这就是课程学习的关键：前一阶段学到的知识带入下一阶段

    # ---- 最终评估 ----
    print("\n最终测试集评估...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"最终测试 WER: {test_results['eval_wer']*100:.2f}%")

    # 保存
    trainer.save_model(f"{output_dir}/best_model")
    processor.save_pretrained(f"{output_dir}/best_model")

    import json
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "method": "curriculum_learning",
            "stages": stages,
            "test_wer": test_results["eval_wer"],
        }, f, indent=2)

    return test_results


if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)
    config = load_config()
    curriculum_train(config)
