$script:ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$env:ANJU_KWS_PYTHON = "D:\conda-envs\cosyvoice310\python.exe"
$env:PYTHONPATH = @(
    (Join-Path $script:ProjectRoot "src"),
    (Join-Path $script:ProjectRoot "third_party\wekws")
) -join ";"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path -LiteralPath $env:ANJU_KWS_PYTHON)) {
    throw "Configured Python was not found: $env:ANJU_KWS_PYTHON"
}
