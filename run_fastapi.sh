#!/bin/bash

export PYTHONPATH=third_party/Matcha-TTS
conda activate cosyvoice

# nohup uvicorn main:app --port 6006 &
nohup uvicorn main:app --host 0.0.0.0 --port 6006 --workers 6 &