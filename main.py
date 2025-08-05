import io,time
from fastapi import FastAPI, Response,File, UploadFile,Form
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pydantic import BaseModel
from typing import Optional
import json
import os
import numpy as np
import torch
from torchaudio.transforms import Resample


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT')
# sft usage
# print(cosyvoice.list_avaliable_spks())
app = FastAPI()

def read_lab_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(lines[0])
    return lines[0]

def buildResponse(output):
    buffer = io.BytesIO()
    torchaudio.save(buffer, output, 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")


class RecommendationRequest(BaseModel):
    text: str
    role_name: Optional[str] = None
    reference_audio: Optional[str] = None
    


# miaoer 嗯小哥哥，这边是一个绿色健康的有偿交友类的平台哦，这边主要是分为游戏、点唱和聊天交友的板块。
# xiaomei 对，大家晚上睡不着的话可以到这里来说说话然后听听歌。
# xiaoqiao 欢迎我我们睿智的勇敢狗，大家都是刚刚到这边玩的是吗？平常喜欢听歌还是玩游戏呢？
# 线上在用不要更改！！！！！
@app.post("/api/voice/tts")
async def tts(recommendation: RecommendationRequest):
    # 直接访问 Pydantic 模型的属性
    input_text = recommendation.text
    role_name = recommendation.role_name
    reference_audio = recommendation.reference_audio
    if not role_name:
        role_name = "miaoer"
    if not reference_audio:
        sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
        sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)
    else:
        sample_text = "role/{role_name}/sample{number}.lab".format(role_name=role_name,number=str(reference_audio))
        sample_wav = "role/{role_name}/sample{number}.wav".format(role_name=role_name,number=str(reference_audio))
    
    start = time.time()
    prompt_speech_16k = load_wav(sample_wav, 16000)
    prompt_text = read_lab_file(sample_text)
    output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)
    end = time.time()
    print("infer time:", end-start)
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

@app.post("/api/voice/tts_pcm")
async def tts_pcm(recommendation: RecommendationRequest):
    # 直接访问 Pydantic 模型的属性
    input_text = recommendation.text
    role_name = recommendation.role_name
    reference_audio = recommendation.reference_audio
    if not role_name:
        role_name = "miaoer"
    if not reference_audio:
        sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
        sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)
    else:
        sample_text = "role/{role_name}/sample{number}.lab".format(role_name=role_name,number=str(reference_audio))
        sample_wav = "role/{role_name}/sample{number}.wav".format(role_name=role_name,number=str(reference_audio))
    
    start = time.time()
    prompt_speech_16k = load_wav(sample_wav, 16000)
    prompt_text = read_lab_file(sample_text)
    output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)
    end = time.time()
    print("infer time:", end-start)
    buffer = io.BytesIO()
    resampler = Resample(orig_freq=22050, new_freq=16000)
    resampled_audio = resampler(output['tts_speech'])
    # int16_audio = resampled_audio.to(torch.int16)
    pcm_data = resampled_audio.numpy().tobytes()
    # torchaudio.save(buffer, resampled_audio, 16000, format="wav",encoding="PCM_S16")
    buffer.seek(0)
    return Response(content=pcm_data, media_type="audio/pcm")

@app.post("/api/inference/role-tts")
async def roleTTS(recommendation: RecommendationRequest):
    # 直接访问 Pydantic 模型的属性
    input_text = recommendation.text
    role_name = recommendation.role_name
    reference_audio = recommendation.reference_audio
    if not role_name:
        role_name = "miaoer"
    if not reference_audio:
        sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
        sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)
    else:
        sample_text = "role/{role_name}/sample{number}.lab".format(role_name=role_name,number=str(reference_audio))
        sample_wav = "role/{role_name}/sample{number}.wav".format(role_name=role_name,number=str(reference_audio))
    
    start = time.time()
    prompt_text = read_lab_file(sample_text)
    prompt_speech = load_wav(sample_wav, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k)
    end = time.process_time()
    print("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])


@app.post("/api/voice/stream")
async def stream(recommendation: RecommendationRequest):
   # 直接访问 Pydantic 模型的属性
    input_text = recommendation.text
    role_name = recommendation.role_name
    reference_audio = recommendation.reference_audio
    if not role_name:
        role_name = "miaoer"
    if not reference_audio:
        sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
        sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)
    else:
        sample_text = "role/{role_name}/sample{number}.lab".format(role_name=role_name,number=str(reference_audio))
        sample_wav = "role/{role_name}/sample{number}.wav".format(role_name=role_name,number=str(reference_audio))
        
    # 处理tts
    start = time.process_time()
    # prompt_speech_16k = load_wav(wav_file_path, 16000)
    prompt_speech = load_wav(sample_wav, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    prompt_text = sample_text
    
    def generate_stream():
        # resample = torchaudio.transforms.Resample(16000, 22050)
        mulaw_encode = torchaudio.transforms.MuLawEncoding()
        for chunk in cosyvoice.stream(input_text, prompt_text, prompt_speech_16k):
            # 将音频数据块转换为 mu-law 编码的音频帧
            # chunk = resample(chunk)
            # chunk = mulaw_encode(chunk)
            # 将音频帧发送给客户端
            yield chunk.numpy().tobytes()
    end = time.process_time()
    print("infer time:", end-start)
    return StreamingResponse(generate_stream(), media_type="audio/pcm")
    

class AudioRequest(BaseModel):
    text: str
    role_name: str

@app.post("/api/voice/zero_shot_tts")
async def upload_audio(file: UploadFile, text: str = Form(...),role_name: str = Form(...),sample_text: str = Form(...)):
    role_dir = "role"
    # 检查 role 目录下是否存在 role_name 文件夹
    if not os.path.exists(os.path.join(role_dir, role_name)):
        # 如果不存在，则创建一个
        os.mkdir(os.path.join(role_dir, role_name))

    # 保存上传的音频文件
    wav_file_path = os.path.join(role_dir, role_name, "sample.wav")
    with open(wav_file_path, "wb") as f:
        f.write(await file.read())
       
    # 将 text 值保存到 sample.lab 文件中
    lab_file_path = os.path.join(role_dir, role_name, "sample.lab")
    with open(lab_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    # 处理tts
    start = time.process_time()
    prompt_speech_16k = load_wav(wav_file_path, 16000)
    prompt_text = sample_text
    output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)

    end = time.process_time()
    print("infer time:", end-start)
    buffer = io.BytesIO()
    torchaudio.save(buffer, output['tts_speech'], 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")




@app.get("/")
async def root():
    return {"message": "Hello World"}