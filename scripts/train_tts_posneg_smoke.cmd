@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 '%~dp0..\third_party\wekws\wekws\bin\train.py' --gpus 0 --ddp.dist_backend gloo --config '%~dp0..\configs\train\fsmn_ctc_smoke_posneg.yaml' --train_data '%~dp0..\data\prepared_tts_pos500_neg1000\train\data.list' --cv_data '%~dp0..\data\prepared_tts_pos500_neg1000\dev\data.list' --model_dir '%~dp0..\experiments\smoke_tts_pos500_neg1000_001' --tensorboard_dir '%~dp0..\experiments\tensorboard' --dict '%~dp0..\dict\tts_pos500_neg1000' --num_keywords 6 --min_duration 5 --num_workers 1 --prefetch 2"
exit /b %ERRORLEVEL%
