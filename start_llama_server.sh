#!/usr/bin/env bash
# MiniMax H3 Prompter - llama-server (Qwen3.8-27B, unsloth GGUF)
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
MODEL_ARGS="${MODEL_ARGS:--hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL}"
# Local files instead:
# MODEL_ARGS="-m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj /models/mmproj-F16.gguf"
exec "$LLAMA_SERVER" $MODEL_ARGS \
  --jinja -ngl 999 -c 32768 -np 1 --no-mmap \
  --host 127.0.0.1 --port 8080 --alias qwen3.8-27b
