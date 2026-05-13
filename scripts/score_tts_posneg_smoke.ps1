$ErrorActionPreference = "Stop"
. (Join-Path (Split-Path -Parent $PSScriptRoot) "configs\env.ps1")

$config = Join-Path $script:ProjectRoot "experiments\smoke_tts_pos500_neg1000_001\config.yaml"
$testData = Join-Path $script:ProjectRoot "data\prepared_tts_pos500_neg1000\test\data.list"
$dictDir = Join-Path $script:ProjectRoot "dict\tts_pos500_neg1000"
$checkpoint = Join-Path $script:ProjectRoot "experiments\smoke_tts_pos500_neg1000_001\final.pt"
$scoreFile = Join-Path $script:ProjectRoot "experiments\smoke_tts_pos500_neg1000_001\score.txt"
$scoreScript = Join-Path $script:ProjectRoot "third_party\wekws\wekws\bin\score_ctc.py"
$keyword = "\u5b89\u5c45\u5c0f\u5b9d"

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
