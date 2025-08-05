import io,time
from fastapi import FastAPI, Response,File, UploadFile,Form,HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from pydantic import BaseModel
from typing import Optional
import json
import os
import numpy as np
import torch
from torchaudio.transforms import Resample
import sys
from fastapi.responses import StreamingResponse
import load_roles


sys.path.append('third_party/Matcha-TTS')
global cosyvoice
# cosyvoice = CosyVoice2(
#     'pretrained_models/CosyVoice2-0.5B',
#     load_jit=False,   # 如果已转 TensorRT，无需启用 JIT
#     load_trt=True,     # 启用 TensorRT 加速（需转换后的引擎文件）
#     fp16=True          # 启用 FP16 加速
# )
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,   # 如果已转 TensorRT，无需启用 JIT
    load_trt=True,     # 启用 TensorRT 加速（需转换后的引擎文件）
    fp16=True          # 启用 FP16 加速
)
print("加载模型成功")
global load_roles
roles_data = load_roles.load_role_data(role_dir="role")
print("加载角色数据成功")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        
              
# @app.middleware("http")
# async def token_validator(request: Request, call_next):
#     if request.method == "POST":
#         token = request.headers.get("x-api-token")
#         if token != "6006":
#             raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
#     return await call_next(request)
              

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

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

class RecommendationRequest(BaseModel):
    text: str
    role_name: Optional[str] = None
    reference_audio: Optional[str] = None
    
# 添加多线程 异步
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# 线上在用不要更改！！！！！
@app.post("/api/voice/tts")
async def tts(recommendation: RecommendationRequest):
    try:
        # 直接访问 Pydantic 模型的属性
        input_text = recommendation.text
        role_name = recommendation.role_name
        reference_audio = recommendation.reference_audio
        start = time.time()
        if not role_name:
            role_name = "miaoer"
        if not reference_audio:
            sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
            sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)
        else:
            sample_text = "role/{role_name}/sample{number}.lab".format(role_name=role_name,number=str(reference_audio))
            sample_wav = "role/{role_name}/sample{number}.wav".format(role_name=role_name,number=str(reference_audio))
        # 验证文件存在性
        if not os.path.exists(sample_text):
            raise HTTPException(404, detail=f"角色音色不存在: {role_name}")
        if not os.path.isfile(sample_text):  # 确保是文件
            raise HTTPException(400, detail=f"角色音色不存在: {role_name}")
        
        prompt_speech_16k = load_wav(sample_wav, 16000)
        prompt_text = read_lab_file(sample_text)
        # output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=False)
        print("加载音频时间 time:", time.time()-start)

        buffer = io.BytesIO()
        for i, j in enumerate(cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=False)):
        # torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
            torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")
        buffer.seek(0)
        end = time.time()
        print("模型推理时间infer time:", end-start)
        return Response(content=buffer.read(-1), media_type="audio/wav")
    except HTTPException as he:
        # 已经处理过的异常直接抛出
        raise he
    except Exception as e:
        # 处理其他未捕获的异常
        raise HTTPException(
            status_code=400,
            detail=f"Internal server error: {str(e)}"
        )
        
# 线上在用不要更改！！！！！
@app.post("/api/voice/tts_v2")
async def tts(recommendation: RecommendationRequest):
    try:
        # 直接访问 Pydantic 模型的属性
        input_text = recommendation.text
        role_name = recommendation.role_name
        reference_audio = recommendation.reference_audio
        if not role_name:
            role_name = "miaoer"
        start = time.time()
        if not roles_data.get(role_name):
            raise HTTPException(404, detail=f"角色音色不存在: {role_name}")
        prompt_speech_16k = roles_data[role_name]["default"]["wav"]
        prompt_text = roles_data[role_name]["default"]["text"]
        buffer = io.BytesIO()
        for i, j in enumerate(cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=False)):
            torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")
        buffer.seek(0)
        end = time.time()
        print("infer time:", end-start)
        return Response(content=buffer.read(-1), media_type="audio/wav")
    except HTTPException as he:
        # 已经处理过的异常直接抛出
        raise he
    except Exception as e:
        # 处理其他未捕获的异常
        raise HTTPException(
            status_code=400,
            detail=f"Internal server error: {str(e)}"
        )
        
@app.get("/api/voice/tts_stream")
@app.post("/api/voice/tts_stream")
async def tts_stream(recommendation: RecommendationRequest):
    try:
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
        # 验证文件存在性
        if not os.path.exists(sample_text):
            raise HTTPException(404, detail=f"角色音色不存在: {role_name}")
        if not os.path.isfile(sample_text):  # 确保是文件
            raise HTTPException(400, detail=f"角色音色不存在: {role_name}")
        start = time.time()
        prompt_speech_16k = load_wav(sample_wav, 16000)
        prompt_text = read_lab_file(sample_text)
        model_output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=True)
        return StreamingResponse(generate_data(model_output))

    except HTTPException as he:
        # 已经处理过的异常直接抛出
        raise he
    except Exception as e:
        # 处理其他未捕获的异常
        raise HTTPException(
            status_code=400,
            detail=f"Internal server error: {str(e)}"
        )

class InstructRequest(BaseModel):
    text: str
    role_name: Optional[str] = None
    instruct_text: Optional[str] = None
    
@app.post("/api/voice/tts_instruct")
async def tts(recommendation: InstructRequest):
    # 直接访问 Pydantic 模型的属性
    input_text = recommendation.text
    role_name = recommendation.role_name
    instruct_text = recommendation.instruct_text
    if not role_name:
        role_name = "miaoer"

    sample_text = "role/{role_name}/sample.lab".format(role_name=role_name)
    sample_wav = "role/{role_name}/sample.wav".format(role_name=role_name)

    
    start = time.time()
    prompt_speech_16k = load_wav(sample_wav, 16000)
    prompt_text = read_lab_file(sample_text)
    # output = cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=False)
    
    
    buffer = io.BytesIO()
    if '' != instruct_text:
        print('用instruct_text：'+instruct_text)
        for i, j in enumerate(cosyvoice.inference_instruct2(input_text, instruct_text, prompt_speech_16k, stream=False)):
            torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")
    else:
        print("不用")
        for i, j in enumerate(cosyvoice.inference_zero_shot(input_text, prompt_text, prompt_speech_16k,stream=False)):
            torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")
    
    buffer.seek(0)
    end = time.time()
    print("infer time:", end-start)
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