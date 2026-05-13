@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0..\configs\env.ps1'; & $env:ANJU_KWS_PYTHON '%~dp0..\src\anju_kws\data\segment_noise_recording.py' --input_wav '%~dp0..\data\rk3566_noise_recordings\recording_20260507085901_16k.wav' --output_dir '%~dp0..\data\raw\rk3566_negative\noise\rkneg_noise_20260507085901_segments' --manifest '%~dp0..\data\manifests\rk3566_noise_20260507085901_segments.jsonl' --sample_id_prefix 'rkneg_noise_20260507085901' --source_remote_path '/storage/emulated/0/Records/recording_20260507085901.m4a' --segment_sec 5 --min_last_sec 2"
exit /b %ERRORLEVEL%
