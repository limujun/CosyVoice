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
import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
import time

def main():
    url = "https://u176130-b26f-536cbf2a.westc.gpuhub.com:8443/api/voice/tts"
    
    payload = {
        "text": "你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？",
        "role_name": "xiaoqiao",
        "reference_audio": ""
    } 
    # files = [('prompt_wav', ('prompt_wav', open(args.prompt_wav, 'rb'), 'application/octet-stream'))]
    response = requests.post(url, json=payload, stream=False)
    print(response)
    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    # print('save response to {}'.format(args.tts_wav))
    torchaudio.save("demo.wav", tts_speech, target_sr)
    print('get response')


if __name__ == "__main__":
    prompt_sr, target_sr = 16000, 22050
    start_time = time.time()
    main()
    print("运行时间："+str(time.time()-start_time))
