$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot "configs\env.ps1")

$expDir = Join-Path $projectRoot "experiments\smoke_tts_cleanpos200_rk20min_noisy_main_15ep_001"
$testData = Join-Path $projectRoot "data\prepared_tts_cleanpos200_rk20min_noisy_main\test\data.list"
$dictDir = Join-Path $projectRoot "dict\tts_cleanpos200_rk20min_noisy_main"
$scoreScript = Join-Path $projectRoot "third_party\wekws\wekws\bin\score_ctc.py"
$keyword = "\u5b89\u5c45\u5c0f\u5b9d"

$runs = @(
    @{ Checkpoint = "best.pt"; ScoreFile = "score_test_best.txt" },
    @{ Checkpoint = "final.pt"; ScoreFile = "score_test_final.txt" }
)

foreach ($run in $runs) {
    & $env:ANJU_KWS_PYTHON $scoreScript `
        --config (Join-Path $expDir "config.yaml") `
        --test_data $testData `
        --dict $dictDir `
        --gpu 0 `
        --checkpoint (Join-Path $expDir $run.Checkpoint) `
        --batch_size 16 `
        --num_workers 1 `
        --prefetch 2 `
        --score_file (Join-Path $expDir $run.ScoreFile) `
        --keywords $keyword `
        --token_file (Join-Path $dictDir "dict.txt")
}
