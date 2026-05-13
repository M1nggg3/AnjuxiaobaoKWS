@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update_wekws_code.ps1" %*
exit /b %ERRORLEVEL%
