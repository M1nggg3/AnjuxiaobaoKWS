$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSScriptRoot) "configs\env.ps1")

$expDir = Join-Path $script:ProjectRoot "experiments\smoke_tts_softpos_oversample_15ep_001"
$config = Join-Path $expDir "config.yaml"
$testData = Join-Path $script:ProjectRoot "data\prepared_tts_clean_and_softpos_noisy_oversample\test\data.list"
$dictDir = Join-Path $script:ProjectRoot "dict\tts_clean_and_softpos_noisy_oversample"
$scoreScript = Join-Path $script:ProjectRoot "third_party\wekws\wekws\bin\score_ctc.py"
$keyword = "\u5b89\u5c45\u5c0f\u5b9d"

$runs = @(
    @{ Checkpoint = "8.pt"; ScoreFile = "score_test_epoch8_best.txt" },
    @{ Checkpoint = "final.pt"; ScoreFile = "score_test_final.txt" }
)

foreach ($run in $runs) {
    $checkpoint = Join-Path $expDir $run.Checkpoint
    $scoreFile = Join-Path $expDir $run.ScoreFile

    & $env:ANJU_KWS_PYTHON $scoreScript `
        --config $config `
        --test_data $testData `
        --dict $dictDir `
        --gpu 0 `
        --checkpoint $checkpoint `
        --batch_size 16 `
        --num_workers 1 `
        --prefetch 2 `
        --score_file $scoreFile `
        --keywords $keyword `
        --token_file (Join-Path $dictDir "dict.txt")
}
