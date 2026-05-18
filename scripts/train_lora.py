"""
阶段4：LoRA微调
- 用peft库给Whisper加LoRA旁路
- 只训练约1%的参数
- 这是项目的核心训练脚本
"""

import yaml
import torch
from dataclasses import dataclass
import gc
from typing import Any, Dict, List, Union

from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 数据整理器：把一个batch的数据padding到相同长度
# 面试可能会问：为什么需要这个？
# 因为每条音频长度不同，需要padding到batch内最长的那个
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # 处理输入特征（音频）
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 处理标签（文本token ids）
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # padding token替换为-100，这样计算loss时会忽略它们
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 去掉开头的BOS token（Whisper的decoder会自动加）
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_wer(pred, processor):
    """计算WER的回调函数"""
    wer_metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100替换回pad_token_id，方便decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 把每个字用空格隔开，让 WER 计算器当 CER 用
    pred_str = [" ".join(list(s.replace(" ", ""))) for s in pred_str]
    label_str = [" ".join(list(s.replace(" ", ""))) for s in label_str]
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    # 此时的 wer 实际上就是 CER
    return {"wer": wer}


def train_lora(config, lora_rank=None):
    """
    LoRA微调主函数

    参数:
        config: 配置字典
        lora_rank: 可选，用于rank消融实验时覆盖配置中的rank值
    """

    rank = lora_rank or config["lora"]["r"]

    print("=" * 60)
    print(f"阶段4：LoRA 微调 (rank={rank})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. 加载模型 ----
    model_name = config["model"]["name"]
    print(f"\n[1] 加载基座模型: {model_name}")

    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=config["model"]["language"],
        task=config["model"]["task"],
    )

    # 强制decoder使用中文
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=config["model"]["language"],
        task=config["model"]["task"],
    )
    model.config.suppress_tokens = []

    # ---- 2. 添加LoRA ----
    print(f"\n[2] 添加 LoRA (rank={rank})...")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出类似: trainable params: 2,359,296 || all params: 249,441,280 || trainable%: 0.95%

    # ---- 3. 加载数据 ----
    print("\n[3] 加载数据集...")
    dataset = load_from_disk("/root/autodl-tmp/data/whisper_dataset")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ---- 4. 训练参数 ----
    output_dir = f"./outputs/lora_rank{rank}"
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
        logging_dir=config["training"]["logging_dir"],
        logging_steps=50,
        predict_with_generate=True,      # 评估时用generate而不是forward
        generation_max_length=225,
        report_to="tensorboard",
    )

    # ---- 5. 开始训练 ----
    print("\n[4] 开始训练...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_wer(pred, processor),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # ---- 6. 保存 ----
    print(f"\n[5] 保存模型到 {output_dir}/best_model")
    trainer.save_model(f"{output_dir}/best_model")
    processor.save_pretrained(f"{output_dir}/best_model")

    # ---- 7. 最终评估 ----
    print("\n[6] 在测试集上最终评估...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"  测试集 WER: {test_results['eval_wer']*100:.2f}%")

    # 保存结果
    import json
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "rank": rank,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "test_wer": test_results["eval_wer"],
        }, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return test_results


if __name__ == "__main__":
    config = load_config()
    train_lora(config)
