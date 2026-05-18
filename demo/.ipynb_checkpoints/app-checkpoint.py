"""
阶段6：Gradio Demo
- 可部署到 HuggingFace Spaces，面试时直接分享链接
- 支持麦克风录音和文件上传两种模式
"""

import time
import gradio as gr
from faster_whisper import WhisperModel


def load_model():
    """加载模型（只加载一次）"""
    print("加载 faster-whisper 模型...")
    model = WhisperModel(
        "./outputs/faster_whisper_model",
        device="cuda",          # HuggingFace Spaces用 "cpu"
        compute_type="int8",
    )
    print("模型加载完成")
    return model


MODEL = load_model()


def transcribe(audio_path):
    """
    转写音频文件

    参数:
        audio_path: Gradio传入的音频文件路径

    返回:
        转写文本 + 性能指标
    """
    if audio_path is None:
        return "请上传音频或使用麦克风录音", ""

    # 推理
    start = time.perf_counter()
    segments, info = MODEL.transcribe(audio_path, language="zh",beam_size = 1)
    segments_list = list(segments)
    inference_time = time.perf_counter() - start

    # 拼接文本
    text = ""
    for seg in segments_list:
        text += f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}\n"

    # 性能信息
    audio_duration = info.duration
    rtf = inference_time / audio_duration if audio_duration > 0 else 0

    stats = (
        f"音频时长: {audio_duration:.1f}s\n"
        f"推理耗时: {inference_time:.3f}s\n"
        f"RTF: {rtf:.4f}\n"
        f"{'✅ 可实时处理' if rtf < 1 else '❌ 无法实时处理'}"
    )

    return text.strip(), stats


# ---- 构建界面 ----
with gr.Blocks(title="Whisper ASR Demo") as demo:

    gr.Markdown(
        """
        # 🎙️ 噪声鲁棒语音识别系统
        基于 Whisper + LoRA 微调，针对嘈杂环境优化

        **特点：**
        - LoRA 参数高效微调，可训练参数仅占 ~1%
        - faster-whisper INT8 量化推理
        - 课程学习训练策略，提升噪声鲁棒性
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="上传音频或使用麦克风",
            )
            submit_btn = gr.Button("开始识别", variant="primary")

        with gr.Column():
            text_output = gr.Textbox(label="识别结果", lines=10)
            stats_output = gr.Textbox(label="性能指标", lines=5)

    submit_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[text_output, stats_output],
    )

    gr.Markdown(
        """
        ---
        **技术栈：** Whisper-small + LoRA (PEFT) + faster-whisper (CTranslate2 INT8)

        **实验结果：**
        | 方案 | WER | 推理RTF | 加速比 |
        |------|-----|---------|------|
        | Whisper原版 | 28% | 0.29 | 1x |
        | + LoRA微调 | 25.3% | 0.29 | 1x |
        | + INT8量化 | 25.3% | 0.017 | **24x** |
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
