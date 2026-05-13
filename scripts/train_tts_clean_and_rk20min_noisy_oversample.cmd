@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; $env:LOCAL_RANK='0'; $env:WORLD_SIZE='1'; & $env:ANJU_KWS_PYTHON '%~dp0..\third_party\wekws\wekws\bin\train.py' --config '%~dp0..\configs\train\fsmn_ctc_clean_and_rk20min_noisy_oversample.yaml' --train_data '%~dp0..\data\prepared_tts_rk20min_noisy_with_purenoise_oversample\train\data.list' --cv_data '%~dp0..\data\prepared_tts_rk20min_noisy_with_purenoise_oversample\dev\data.list' --model_dir '%~dp0..\experiments\smoke_tts_rk20min_noisy_with_purenoise_15ep_001' --dict '%~dp0..\dict\tts_rk20min_noisy_with_purenoise_oversample' --num_keywords 6 --min_duration 5 --num_workers 1 --prefetch 2"
exit /b %ERRORLEVEL%
