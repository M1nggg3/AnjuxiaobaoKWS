@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON '%~dp0..\src\anju_kws\data\generate_noise_augmented_dataset.py' --positive_manifest 'E:\CodeWorking\Dataset\anju_xiaobao_cosyvoice2_500_gpu\manifest.jsonl' --negative_manifest 'E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu\manifest.jsonl' --noise_manifest '%~dp0..\data\manifests\rk3566_noise_20260507085901_segments.jsonl' --output_dir 'E:\CodeWorking\Dataset\anju_xiaobao_noise_augmented_rk20min_20260507' --num_positive 500 --num_negative 1000 --positive_snr_db '15,20,25,30' --negative_snr_db '5,10,15,20'"
exit /b %ERRORLEVEL%
