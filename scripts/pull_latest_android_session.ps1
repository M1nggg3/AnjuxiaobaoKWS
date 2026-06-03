param(
    [string]$OutDir = "E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\rk3566_live_capture_debug\app_sessions",
    [string]$Package = "cn.org.wenet.wekws"
)

$ErrorActionPreference = "Stop"

function Convert-PcmToWav {
    param(
        [Parameter(Mandatory = $true)][string]$PcmPath,
        [Parameter(Mandatory = $true)][string]$WavPath,
        [int]$SampleRate = 16000
    )

    $pcm = [System.IO.File]::ReadAllBytes($PcmPath)
    $dataSize = $pcm.Length
    $riffSize = 36 + $dataSize
    $byteRate = $SampleRate * 2
    $blockAlign = 2

    $stream = [System.IO.File]::Open($WavPath, [System.IO.FileMode]::Create)
    try {
        $writer = New-Object System.IO.BinaryWriter($stream)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("RIFF"))
        $writer.Write([int]$riffSize)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("WAVE"))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("fmt "))
        $writer.Write([int]16)
        $writer.Write([int16]1)
        $writer.Write([int16]1)
        $writer.Write([int]$SampleRate)
        $writer.Write([int]$byteRate)
        $writer.Write([int16]$blockAlign)
        $writer.Write([int16]16)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes("data"))
        $writer.Write([int]$dataSize)
        $writer.Write($pcm)
        $writer.Flush()
    } finally {
        $stream.Dispose()
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$remoteRoot = "/sdcard/Android/data/$Package/files"
$latestEnhanced = adb shell "ls -t $remoteRoot/captures/listen_session_*_enhanced_16k_s16le.pcm 2>/dev/null | head -n 1"
$latestEnhanced = ($latestEnhanced -join "`n").Trim()
if ($latestEnhanced) {
    $enhancedName = [System.IO.Path]::GetFileNameWithoutExtension($latestEnhanced)
    $sessionName = $enhancedName -replace "_enhanced_16k_s16le$", ""
    $latestLog = "$remoteRoot/logs/$sessionName.log"
} else {
    $latestLog = adb shell "ls -t $remoteRoot/logs/listen_session_*.log 2>/dev/null | head -n 1"
    $latestLog = ($latestLog -join "`n").Trim()
    if (-not $latestLog) {
        throw "No listen_session log found under $remoteRoot/logs"
    }
    $sessionName = [System.IO.Path]::GetFileNameWithoutExtension($latestLog)
}
$localLog = Join-Path $OutDir "$sessionName.log"
adb pull $latestLog $localLog | Out-Null

$remoteRaw = "$remoteRoot/captures/${sessionName}_raw_16k_s16le.pcm"
$remoteEnhanced = "$remoteRoot/captures/${sessionName}_enhanced_16k_s16le.pcm"
$localRaw = Join-Path $OutDir "${sessionName}_raw_16k_s16le.pcm"
$localEnhanced = Join-Path $OutDir "${sessionName}_enhanced_16k_s16le.pcm"

adb pull $remoteRaw $localRaw | Out-Null
adb pull $remoteEnhanced $localEnhanced | Out-Null

$localRawWav = [System.IO.Path]::ChangeExtension($localRaw, ".wav")
$localEnhancedWav = [System.IO.Path]::ChangeExtension($localEnhanced, ".wav")
Convert-PcmToWav -PcmPath $localRaw -WavPath $localRawWav
Convert-PcmToWav -PcmPath $localEnhanced -WavPath $localEnhancedWav

[PSCustomObject]@{
    Session = $sessionName
    Log = $localLog
    RawPcm = $localRaw
    RawWav = $localRawWav
    EnhancedPcm = $localEnhanced
    EnhancedWav = $localEnhancedWav
}
