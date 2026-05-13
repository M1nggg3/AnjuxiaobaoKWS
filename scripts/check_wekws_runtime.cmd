@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_wekws_runtime.ps1" %*
exit /b %ERRORLEVEL%
