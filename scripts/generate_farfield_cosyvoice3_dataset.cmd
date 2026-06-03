@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=E:\CodeWorking\Project\AnJuXiaoBaoKWS\src;E:\CodeWorking\Project\AnJuXiaoBaoKWS;E:\CodeWorking\Project\CosyVoice;E:\CodeWorking\Project\CosyVoice\third_party\Matcha-TTS;%PYTHONPATH%

"D:\conda-envs\cosyvoice310\python.exe" -m anju_kws.tts.generate_farfield_cosyvoice3_dataset %*

endlocal
