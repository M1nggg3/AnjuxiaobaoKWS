@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON '%~dp0..\src\anju_kws\data\prepare_tts_smoke.py' --source_dir 'E:\CodeWorking\Dataset\anju_xiaobao_cosyvoice2_500_gpu' --output_dir '%~dp0..\data\prepared_tts_smoke' --dict_dir '%~dp0..\dict\tts_smoke'"
exit /b %ERRORLEVEL%
