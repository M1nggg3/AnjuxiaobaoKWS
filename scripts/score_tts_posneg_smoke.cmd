@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0score_tts_posneg_smoke.ps1"
exit /b %ERRORLEVEL%
