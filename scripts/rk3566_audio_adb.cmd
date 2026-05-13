@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PS_SCRIPT=%SCRIPT_DIR%rk3566_audio_adb.ps1

if "%~1"=="" goto :usage

set ACTION=%~1
shift /1

if /I "%ACTION%"=="test" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%PS_SCRIPT%'; Test-RkAdb"
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="list" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%PS_SCRIPT%'; Get-RkAudioFiles %* | Format-Table -AutoSize"
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="manifest" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%PS_SCRIPT%'; Save-RkAudioManifest %*"
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="pull" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%PS_SCRIPT%'; Pull-RkAudioFiles %*"
  exit /b %ERRORLEVEL%
)

if /I "%ACTION%"=="record" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". '%PS_SCRIPT%'; Start-RkTinycapRecord %*"
  exit /b %ERRORLEVEL%
)

:usage
echo Usage:
echo   scripts\rk3566_audio_adb.cmd test
echo   scripts\rk3566_audio_adb.cmd list -RemoteRoots /sdcard,/data/local/tmp
echo   scripts\rk3566_audio_adb.cmd manifest -RemoteRoots /sdcard -OutFile .\data\rk3566_audio\manifest.csv
echo   scripts\rk3566_audio_adb.cmd pull -RemoteRoots /sdcard -LocalRoot .\data\rk3566_audio
echo   scripts\rk3566_audio_adb.cmd record -RemoteFile /sdcard/anjuxiaobao_test.wav -Seconds 5
echo.
echo If adb.exe is not in PATH, set ADB_PATH first:
echo   set ADB_PATH=C:\Android\platform-tools\adb.exe
exit /b 1
