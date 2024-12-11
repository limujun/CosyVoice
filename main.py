# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from pydantic import BaseModel
from typing import Optional
import time
from io import BytesIO
from scipy.io.wavfile import write
import torchaudio
import torch
from fastapi import Response,HTTPException
import random



app = FastAPI()
# set cross region allowance
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# 定义请求模型
class TTSRequest(BaseModel):
    text: str
    role_name: str
    reference_audio: str = None


def read_lab_file(file_path):
    """读取 .lab 文件的第一行"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    print(first_line)
    return first_line


@app.post("/api/voice/tts")
async def tts(request: TTSRequest):
    """处理 TTS 请求并返回合成音频流"""
    
    # 获取输入数据
    input_text = request.text
    role_name = request.role_name or "miaoer"
    reference_audio = request.reference_audio or ""
    
    # 如果没有输入文本，提前返回错误响应
    if not input_text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    # 构造音频和文本文件路径
    sample_text = f"role/{role_name}/sample{reference_audio}.lab" if reference_audio else f"role/{role_name}/sample.lab"
    sample_wav = f"role/{role_name}/sample{reference_audio}.wav" if reference_audio else f"role/{role_name}/sample.wav"
    
    # 异步加载提示音频和文本（假设load_wav和read_lab_file支持异步）
    start_time = time.time()
    # prompt_speech_16k = load_wav(sample_wav, 16000) 
    # prompt_text = read_lab_file(sample_text)
    try:
        prompt_speech_16k = load_wav(sample_wav, 16000)
        prompt_text = read_lab_file(sample_text)
    except Exception:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    print("加载音频时间:", time.time() - start_time)
    
    output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)

    # 合并音频数据并转换为 int16 格式
    audio_array = np.concatenate([i['tts_speech'].numpy() for i in output], axis=0)
    audio_array = (audio_array * (2 ** 15)).astype(np.int16)  # 转换为 int16
    tts_speech = torch.from_numpy(audio_array).unsqueeze(0)
    if tts_speech.ndimension() == 3:
        tts_speech = tts_speech[0]  # 选择第一个音频样本

    # 保存张量为音频流并返回
    with BytesIO() as buffer:
        torchaudio.save(buffer, tts_speech, sample_rate=22050, format="wav")
        buffer.seek(0)
        audio_response = buffer.read()

    print("总处理时间:", time.time() - start_time)
    del output, tts_speech  # 手动删除不再使用的变量
    torch.cuda.empty_cache()  # 清理显存
    return Response(content=audio_response, media_type="audio/wav")

@app.post("/api/voice/tts2")
async def tts2(request: TTSRequest):
    """处理 TTS 请求并返回合成音频流"""
    
    # 获取输入数据
    input_text = request.text
    role_name = request.role_name or "xiaoqiao2"
    reference_audio = request.reference_audio or ""
    
    # 如果没有输入文本，提前返回错误响应
    if not input_text:
        raise HTTPException(status_code=400, detail="Text input is required.")
    
    # 指定文件夹路径
    folder_path = "role/{role_name}".format(role_name=role_name)

    # 获取文件夹中的文件数量
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav") and os.path.isfile(os.path.join(folder_path, f))]
    # 如果文件夹中有 .wav 文件
    if wav_files:
        # 随机选择一个文件名并去除扩展名
        random_file = random.choice(wav_files)
        random_file_name = os.path.splitext(random_file)[0]  # 去除 .wav 扩展名
        print(f"随机选择的文件名：{random_file_name}")
    else:
        print("文件夹中没有 .wav 文件。")
    print(f"调用角色{role_name}的{random_file_name}音色")
    # 构造音频和文本文件路径
    sample_text = f"role/{role_name}/{random_file_name}.lab" if reference_audio else f"role/{role_name}/sample.lab"
    sample_wav = f"role/{role_name}/{random_file_name}.wav" if reference_audio else f"role/{role_name}/sample.wav"
    # 异步加载提示音频和文本（假设load_wav和read_lab_file支持异步）
    start_time = time.time()
    try:
        prompt_speech_16k = load_wav(sample_wav, 16000)
        prompt_text = read_lab_file(sample_text)
    except Exception:
        raise HTTPException(status_code=404, detail="Audio file not found.")
    print("加载音频时间:", time.time() - start_time)
    
    output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)

    # 合并音频数据并转换为 int16 格式
    audio_array = np.concatenate([i['tts_speech'].numpy() for i in output], axis=0)
    audio_array = (audio_array * (2 ** 15)).astype(np.int16)  # 转换为 int16
    tts_speech = torch.from_numpy(audio_array).unsqueeze(0)
    if tts_speech.ndimension() == 3:
        tts_speech = tts_speech[0]  # 选择第一个音频样本

    # 保存张量为音频流并返回
    with BytesIO() as buffer:
        torchaudio.save(buffer, tts_speech, sample_rate=22050, format="wav")
        buffer.seek(0)
        audio_response = buffer.read()

    print("总处理时间:", time.time() - start_time)
    del output, tts_speech  # 手动删除不再使用的变量
    torch.cuda.empty_cache()  # 清理显存
    return Response(content=audio_response, media_type="audio/wav")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=6006)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir,load_jit=True,fp16=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
