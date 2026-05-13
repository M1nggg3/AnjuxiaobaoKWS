@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
set "PYTHON=D:\conda-envs\cosyvoice310\python.exe"
set "OUT_DIR=E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu"
set "PROMPT_MANIFEST=E:\CodeWorking\Dataset\anju_xiaobao_cosyvoice2_500_gpu\prompt_voices.jsonl"
set "MODEL_DIR=D:\models\CosyVoice2-0.5B"
set "COSYVOICE_REPO=D:\codeWorking\TTS\CosyVoice"

set "PYTHONPATH=%PROJECT_ROOT%\src;%COSYVOICE_REPO%;%COSYVOICE_REPO%\third_party\Matcha-TTS"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%PYTHON%" (
  echo Python not found: %PYTHON%
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

"%PYTHON%" -m anju_kws.tts.generate_negative_text_plan ^
  --prompt_manifest "%PROMPT_MANIFEST%" ^
  --output_dir "%OUT_DIR%"
if errorlevel 1 exit /b %ERRORLEVEL%

"%PYTHON%" -m anju_kws.tts.generate_negative_cosyvoice2 ^
  --text_plan "%OUT_DIR%\text_plan.jsonl" ^
  --output_dir "%OUT_DIR%" ^
  --model_dir "%MODEL_DIR%" ^
  --cosyvoice_repo "%COSYVOICE_REPO%" ^
  --overwrite ^
  1> "%OUT_DIR%\generation.out.log" ^
  2> "%OUT_DIR%\generation.err.log"

exit /b %ERRORLEVEL%
