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
import json

def main():
    url = "https://cv-ai.lyg.live/api/voice/tts_stream"

    payload = {
      "text": "祝福如果过于保守呢，就不如不祝福。",
      "role_name": "tide",
      "reference_audio": ""
    } 
    # 将请求参数转换为 JSON 格式
    data = json.dumps(payload)

    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, data=data, headers=headers, stream=True)

    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    # print(tts_audio)
    tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    torchaudio.save('test.wav', tts_speech, target_sr)
    logging.info('get response')


if __name__ == "__main__":
    prompt_sr, target_sr = 16000, 22050
    main()
