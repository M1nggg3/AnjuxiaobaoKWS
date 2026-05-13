<#
Utilities for accessing audio files recorded on an RK3566 Android board.

Usage:
  . .\scripts\rk3566_audio_adb.ps1
  Test-RkAdb
  Get-RkAudioFiles -RemoteRoots /sdcard,/data/local/tmp
  Save-RkAudioManifest -RemoteRoots /sdcard -OutFile .\data\rk3566_audio\manifest.csv
  Pull-RkAudioFiles -RemoteRoots /sdcard -LocalRoot .\data\rk3566_audio

If adb.exe is not in PATH:
  . .\scripts\rk3566_audio_adb.ps1 -AdbPath C:\Android\platform-tools\adb.exe
#>

param(
    [string]$AdbPath = $env:ADB_PATH,
    [string]$Serial = $env:ADB_SERIAL
)

if ([string]::IsNullOrWhiteSpace($AdbPath)) {
    $AdbPath = "adb"
}

$script:RkAdbPath = $AdbPath
$script:RkAdbSerial = $Serial

function Resolve-RkAdb {
    if (Test-Path -LiteralPath $script:RkAdbPath) {
        return (Resolve-Path -LiteralPath $script:RkAdbPath).Path
    }

    $cmd = Get-Command $script:RkAdbPath -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $commonPaths = @(
        "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
        "$env:USERPROFILE\AppData\Local\Android\Sdk\platform-tools\adb.exe",
        "$env:ProgramFiles\Android\platform-tools\adb.exe",
        "C:\platform-tools\adb.exe",
        "D:\platform-tools\adb.exe",
        "E:\platform-tools\adb.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path -LiteralPath $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }

    throw "adb.exe was not found. Install Android Platform Tools or pass -AdbPath C:\path\to\adb.exe."
}

function Invoke-RkAdb {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $adb = Resolve-RkAdb
    $args = @()
    if (-not [string]::IsNullOrWhiteSpace($script:RkAdbSerial)) {
        $args += @("-s", $script:RkAdbSerial)
    }
    $args += $Arguments

    & $adb @args
    if ($LASTEXITCODE -ne 0) {
        throw "adb failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

function ConvertTo-RkShellQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Invoke-RkShell {
    param([Parameter(Mandatory = $true)][string]$Command)
    Invoke-RkAdb shell $Command
}

function Test-RkAdb {
    $adb = Resolve-RkAdb
    Write-Host "ADB: $adb"
    Invoke-RkAdb devices
    Write-Host ""
    Write-Host "Board info:"
    Invoke-RkShell "getprop ro.product.model; getprop ro.hardware; uname -a"
}

function Get-RkAudioFiles {
    param(
        [string[]]$RemoteRoots = @("/sdcard", "/data/local/tmp"),
        [string[]]$Extensions = @("wav", "pcm", "mp3", "m4a", "aac", "flac", "ogg", "amr")
    )

    $pattern = "\.(" + (($Extensions | ForEach-Object { [regex]::Escape($_) }) -join "|") + ")$"

    foreach ($root in $RemoteRoots) {
        $quotedRoot = ConvertTo-RkShellQuoted $root
        $command = "if [ -d $quotedRoot ]; then find $quotedRoot -type f 2>/dev/null; fi"
        $files = Invoke-RkShell $command

        foreach ($file in $files) {
            $remotePath = ($file -replace "`r", "").Trim()
            if ([string]::IsNullOrWhiteSpace($remotePath)) {
                continue
            }
            if ($remotePath -notmatch $pattern) {
                continue
            }

            $quotedFile = ConvertTo-RkShellQuoted $remotePath
            $size = (Invoke-RkShell "stat -c %s $quotedFile 2>/dev/null || echo 0" | Select-Object -First 1)
            $mtime = (Invoke-RkShell "stat -c %y $quotedFile 2>/dev/null || echo ''" | Select-Object -First 1)

            [pscustomobject]@{
                RemotePath = $remotePath
                SizeBytes  = [int64]($size -replace "[^\d]", "")
                Modified   = $mtime
            }
        }
    }
}

function Save-RkAudioManifest {
    param(
        [string[]]$RemoteRoots = @("/sdcard", "/data/local/tmp"),
        [string]$OutFile = ".\data\rk3566_audio\manifest.csv"
    )

    $outDir = Split-Path -Parent $OutFile
    if (-not [string]::IsNullOrWhiteSpace($outDir)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    $items = Get-RkAudioFiles -RemoteRoots $RemoteRoots
    $items | Sort-Object RemotePath | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $OutFile
    Write-Host "Saved manifest: $OutFile"
    return $items
}

function Pull-RkAudioFiles {
    param(
        [string[]]$RemoteRoots = @("/sdcard", "/data/local/tmp"),
        [string]$LocalRoot = ".\data\rk3566_audio",
        [switch]$Flat
    )

    New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
    $items = Get-RkAudioFiles -RemoteRoots $RemoteRoots | Sort-Object RemotePath

    foreach ($item in $items) {
        $remote = $item.RemotePath
        if ($Flat) {
            $local = Join-Path $LocalRoot (Split-Path -Leaf $remote)
        }
        else {
            $relative = $remote.TrimStart("/") -replace "/", "\"
            $local = Join-Path $LocalRoot $relative
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $local) | Out-Null
        }

        Write-Host "Pulling $remote -> $local"
        Invoke-RkAdb pull $remote $local
    }

    Write-Host "Pulled $($items.Count) audio file(s) to $LocalRoot"
    return $items
}

function Start-RkTinycapRecord {
    param(
        [string]$RemoteFile = "/sdcard/anjuxiaobao_test.wav",
        [int]$Seconds = 5,
        [int]$Rate = 16000,
        [int]$Channels = 1,
        [int]$Bits = 16
    )

    $quotedFile = ConvertTo-RkShellQuoted $RemoteFile
    $command = "if command -v tinycap >/dev/null 2>&1; then tinycap $quotedFile -D 0 -d 0 -c $Channels -r $Rate -b $Bits -T $Seconds; elif command -v arecord >/dev/null 2>&1; then arecord -D default -c $Channels -r $Rate -f S16_LE -d $Seconds $quotedFile; else echo 'Neither tinycap nor arecord was found on board'; exit 127; fi"
    Invoke-RkShell $command
    Write-Host "Recorded on board: $RemoteFile"
}
