"""
阶段5：模型导出
- 合并LoRA权重到原始模型（ΔW = BA 合并回 W）
- 转换为faster-whisper格式（CTranslate2 INT8量化）

合并后的模型和原始Whisper结构完全一样，推理零额外开销
"""

import yaml
import os
import shutil
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def export_model(config):
    """
    导出流程：
    1. 加载原始Whisper + LoRA权重
    2. 合并 → 普通Whisper模型
    3. 转换为CTranslate2格式（INT8量化）
    """

    print("=" * 60)
    print("阶段5：模型导出")
    print("=" * 60)

    lora_model_path = f"./outputs/lora_rank{config['lora']['r']}/best_model"
    merged_path = "./outputs/merged_model"
    ct2_path = "./outputs/faster_whisper_model"

    # ---- 1. 加载并合并LoRA ----
    print("\n[1/3] 合并LoRA权重...")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        config["model"]["name"]
    )
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    # 核心操作：W_new = W + B × A
    model = model.merge_and_unload()
    print("  LoRA已合并，模型结构恢复为标准Whisper")

    # 保存合并后的模型
    model.save_pretrained(merged_path)
    processor = WhisperProcessor.from_pretrained(lora_model_path)
    processor.save_pretrained(merged_path)
    print(f"  合并模型已保存到 {merged_path}")

    # ---- 2. 转换为CTranslate2格式 ----
    print("\n[2/3] 转换为 faster-whisper 格式（INT8量化）...")

    # 实际转换命令（需要安装ctranslate2）
    convert_cmd = (
        f"ct2-transformers-converter "
        f"--model {merged_path} "
        f"--output_dir {ct2_path} "
        f"--quantization int8 "
        f"--copy_files tokenizer.json preprocessor_config.json"
    )
    print(f"\n  转换命令：{convert_cmd}")
    ret = os.system(convert_cmd)

    if ret == 0:
        print(f"\n  量化模型已保存到 {ct2_path}")
    else:
        print("\n  自动转换失败，请手动运行上面的命令")
        print("  或者安装: pip install ctranslate2")

    # ---- 3. 对比模型大小 ----
    print("\n[3/3] 模型大小对比:")

    def get_dir_size(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / 1024 / 1024  # MB

    if os.path.exists(merged_path):
        print(f"  合并模型 (FP32): {get_dir_size(merged_path):.1f} MB")
    if os.path.exists(ct2_path):
        print(f"  量化模型 (INT8): {get_dir_size(ct2_path):.1f} MB")


if __name__ == "__main__":
    config = load_config()
    export_model(config)
