$ErrorActionPreference = "Continue"

. (Join-Path (Split-Path -Parent $PSScriptRoot) "configs\env.ps1")

Write-Host "Project root: $script:ProjectRoot"
Write-Host "Python: $env:ANJU_KWS_PYTHON"
& $env:ANJU_KWS_PYTHON --version

Write-Host ""
Write-Host "Dependency check:"
$depCheck = @'
import importlib
mods = [
    "torch", "torchaudio", "yaml", "numpy", "scipy", "tqdm",
    "tensorboardX", "onnx", "onnxruntime", "lmdb", "langid",
    "pypinyin", "wenet", "wekws"
]
failed = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "ok")
        print(f"OK   {name} {version}")
    except Exception as exc:
        failed.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")

try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"FAIL torch cuda check: {type(exc).__name__}: {exc}")

raise SystemExit(1 if failed else 0)
'@
$depCheck | & $env:ANJU_KWS_PYTHON -
$depExit = $LASTEXITCODE

Write-Host ""
Write-Host "Training entry check:"
& $env:ANJU_KWS_PYTHON (Join-Path $script:ProjectRoot "third_party\wekws\wekws\bin\train.py") --help
$trainExit = $LASTEXITCODE

Write-Host ""
Write-Host "Offline CTC inference entry check:"
& $env:ANJU_KWS_PYTHON (Join-Path $script:ProjectRoot "third_party\wekws\wekws\bin\score_ctc.py") --help
$scoreExit = $LASTEXITCODE

Write-Host ""
Write-Host "Stream CTC inference entry check:"
Push-Location (Join-Path $script:ProjectRoot "third_party\wekws\examples\hi_xiaowen\s0")
try {
    & $env:ANJU_KWS_PYTHON (Join-Path $script:ProjectRoot "third_party\wekws\wekws\bin\stream_kws_ctc.py") --help
    $streamExit = $LASTEXITCODE
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Summary:"
Write-Host "dependencies_exit=$depExit"
Write-Host "train_help_exit=$trainExit"
Write-Host "score_ctc_help_exit=$scoreExit"
Write-Host "stream_kws_ctc_help_exit=$streamExit"

if ($depExit -eq 0 -and $trainExit -eq 0 -and $scoreExit -eq 0) {
    Write-Host "Basic WeKWS training/offline inference runtime check passed."
    exit 0
}

Write-Host "Basic WeKWS runtime check failed. See missing dependencies or import errors above."
exit 1
