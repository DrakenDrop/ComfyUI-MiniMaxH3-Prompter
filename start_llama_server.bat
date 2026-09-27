@echo off
REM ===== MiniMax H3 Prompter - llama-server (Qwen3.8-27B, unsloth GGUF) =====
REM Edit LLAMA_DIR to the folder where you extracted the llama.cpp CUDA release.
set LLAMA_DIR=C:\llama.cpp

REM Option A (default): download automatically from Hugging Face (model + mmproj vision).
set MODEL_ARGS=-hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL

REM Option B: already downloaded files. Remove REM on the next line and fix the paths.
REM set MODEL_ARGS=-m D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj D:\models\mmproj-F16.gguf

"%LLAMA_DIR%\llama-server.exe" %MODEL_ARGS% ^
  --jinja ^
  -ngl 999 ^
  -c 32768 ^
  -np 1 ^
  --no-mmap ^
  --host 127.0.0.1 --port 8080 ^
  --alias qwen3.8-27b
pause
