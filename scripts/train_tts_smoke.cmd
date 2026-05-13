@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; $env:LOCALAPPDATA=$env:LOCALAPPDATA; & $env:ANJU_KWS_PYTHON -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 '%~dp0..\third_party\wekws\wekws\bin\train.py' --gpus 0 --ddp.dist_backend gloo --config '%~dp0..\configs\train\fsmn_ctc_smoke.yaml' --train_data '%~dp0..\data\prepared_tts_smoke\train\data.list' --cv_data '%~dp0..\data\prepared_tts_smoke\dev\data.list' --model_dir '%~dp0..\experiments\smoke_tts_001' --tensorboard_dir '%~dp0..\experiments\tensorboard' --dict '%~dp0..\dict\tts_smoke' --num_keywords 6 --min_duration 5 --num_workers 1 --prefetch 2"
exit /b %ERRORLEVEL%
