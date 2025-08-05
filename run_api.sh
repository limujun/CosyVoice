#!/bin/bash

export PYTHONPATH=third_party/Matcha-TTS
conda activate cosyvoice


# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# nohup uvicorn main2:app --port 1234 &
# nohup uvicorn main:app --host 0.0.0.0 --port 6006 --workers 4 > uvicorn.log 2>&1 &
# nohup uvicorn main2:app --host 0.0.0.0 --port 6006 --workers 4 > uvicorn.log 2>&1 &
nohup uvicorn main2:app --host 0.0.0.0 --port 6006 --workers 2 > uvicorn.log 2>&1 &