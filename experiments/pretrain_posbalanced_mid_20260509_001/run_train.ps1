$env:ANJU_KWS_PYTHON='D:\conda-envs\cosyvoice310\python.exe'
$env:PYTHONPATH='E:\CodeWorking\Project\AnJuXiaoBaoKWS\src;E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:LOCAL_RANK='0'
$env:WORLD_SIZE='1'
& D:\conda-envs\cosyvoice310\python.exe E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\wekws\bin\train.py `
  --config E:\CodeWorking\Project\AnJuXiaoBaoKWS\configs\train\fsmn_ctc_pretrain_hardneg_20260509.yaml `
  --train_data E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_posbalanced_mid_20260509\train\data.list `
  --cv_data E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_posbalanced_mid_20260509\dev\data.list `
  --model_dir E:\CodeWorking\Project\AnJuXiaoBaoKWS\experiments\pretrain_posbalanced_mid_20260509_001 `
  --dict E:\CodeWorking\Project\AnJuXiaoBaoKWS\dict\pretrain_posbalanced_mid_20260509 `
  --num_keywords 6 `
  --min_duration 5 `
  --num_workers 1 `
  --prefetch 2 `
  --gpus 0 `
  --checkpoint E:\CodeWorking\Project\AnJuXiaoBaoKWS\experiments\pretrain_posbalanced_mid_20260509_001\partial_pretrain_init.pt
exit $LASTEXITCODE
