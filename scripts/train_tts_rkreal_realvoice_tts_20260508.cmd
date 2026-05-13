@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; $env:LOCAL_RANK='0'; $env:WORLD_SIZE='1'; $base='%~dp0..\experiments\rkreal_finetune_20260508_001\best.pt'; & $env:ANJU_KWS_PYTHON '%~dp0..\third_party\wekws\wekws\bin\train.py' --config '%~dp0..\configs\train\fsmn_ctc_rkreal_realvoice_tts_20260508.yaml' --train_data '%~dp0..\data\prepared_tts_rkreal_realvoice_tts_20260508\train\data.list' --cv_data '%~dp0..\data\prepared_tts_rkreal_realvoice_tts_20260508\dev\data.list' --model_dir '%~dp0..\experiments\rkreal_realvoice_tts_20260508_001' --dict '%~dp0..\dict\tts_rkreal_realvoice_tts_20260508' --num_keywords 6 --min_duration 5 --num_workers 1 --prefetch 2 --gpus 0 --checkpoint $base"
exit /b %ERRORLEVEL%
