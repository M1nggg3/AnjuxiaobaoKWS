@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON '%~dp0..\src\anju_kws\data\generate_noise_augmented_dataset.py' --positive_manifest 'E:\CodeWorking\Dataset\anju_xiaobao_cosyvoice2_500_gpu\manifest.jsonl' --negative_manifest 'E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu\manifest.jsonl' --noise_manifest '%~dp0..\data\manifests\rk3566_noise_20260507032409_segments.jsonl' --output_dir 'E:\CodeWorking\Dataset\anju_xiaobao_noise_augmented_20260507' --num_positive 500 --num_negative 500"
exit /b %ERRORLEVEL%
