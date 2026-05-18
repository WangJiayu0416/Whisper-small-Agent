"""
阶段6：FastAPI 推理服务
- POST /transcribe 上传音频返回转写文本
- 包含推理时间统计

面试时能说：「我封装了REST API，支持生产环境部署」
"""

import time
import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


app = FastAPI(title="Whisper ASR Service")

# 全局模型（服务启动时加载一次）
model = None


@app.on_event("startup")
async def load_model():
    global model
    print("加载模型...")
    model = WhisperModel(
        "./outputs/faster_whisper_model",
        device="cuda",
        compute_type="int8",
    )
    print("模型加载完成")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    上传音频文件，返回转写文本

    请求：POST /transcribe  (multipart/form-data, 字段名: file)
    响应：{
        "text": "转写文本",
        "duration_sec": 音频时长,
        "inference_time_sec": 推理耗时,
        "rtf": 实时率
    }
    """
    # 读取上传的音频
    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    # 如果是多声道，取第一个
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    audio_duration = len(audio) / sr

    # 推理
    start = time.perf_counter()
    segments, info = model.transcribe(audio, language="zh")
    text = " ".join([seg.text for seg in segments])
    inference_time = time.perf_counter() - start

    rtf = inference_time / audio_duration

    return JSONResponse({
        "text": text,
        "duration_sec": round(audio_duration, 2),
        "inference_time_sec": round(inference_time, 3),
        "rtf": round(rtf, 4),
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
