@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON '%~dp0..\src\anju_kws\data\prepare_tts_mixed.py' --config '%~dp0..\configs\data\tts_rkreal_finetune_20260508.yaml'"
exit /b %ERRORLEVEL%
