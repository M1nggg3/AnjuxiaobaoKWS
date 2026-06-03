# 服务器个人目录组织规范

本文档用于约定公用服务器上 `/home/clm` 内的个人项目文件组织方式，重点服务于安居小宝 KWS 项目和后续 CosyVoice3 远场模拟数据生成。

## 目录结构

```text
/home/clm/
├── projects/                         # 项目源码
│   ├── AnJuXiaoBaoKWS/
│   └── CosyVoice/
├── models/                           # 模型权重
│   ├── Fun-CosyVoice3-0.5B/
│   └── AnJuXiaoBaoKWS/
│       ├── wekws/
│       └── prototype_dscnn/
├── datasets/                         # 正式数据集
│   └── AnJuXiaoBaoKWS/
│       ├── data/
│       └── README.md
├── outputs/                          # 试听、smoke、临时推理输出
│   └── AnJuXiaoBaoKWS/
│       ├── smoke/
│       └── inference_debug/
├── runs/                             # 长任务运行记录
│   └── AnJuXiaoBaoKWS/
├── logs/                             # 通用日志
│   └── AnJuXiaoBaoKWS/
├── downloads/                        # 下载中转
├── tmp/                              # 可删除临时文件
└── archive/                          # 旧版本归档
    └── AnJuXiaoBaoKWS/
```

## 初始化

在服务器中进入项目目录后运行：

```bash
cd ~/projects/AnJuXiaoBaoKWS
bash scripts/server_setup_anju_workspace.sh
source ~/.anju_xiaobao_kws_env
```

初始化脚本只创建缺失目录和辅助配置，不移动已有文件，不删除已有文件。

## 环境变量

初始化后会生成：

```bash
~/.anju_xiaobao_kws_env
```

其中包含：

```bash
export ANJU_PROJECT="$HOME/projects/AnJuXiaoBaoKWS"
export COSYVOICE_REPO="$HOME/projects/CosyVoice"
export COSYVOICE3_MODEL="$HOME/models/Fun-CosyVoice3-0.5B"
export ANJU_DATA_ROOT="$HOME/datasets/AnJuXiaoBaoKWS/data"
export ANJU_BASE_DATASET="$ANJU_DATA_ROOT/anju_xiaobao_kws_dataset_20260508"
export ANJU_OUTPUT_ROOT="$HOME/outputs/AnJuXiaoBaoKWS"
export ANJU_RUN_ROOT="$HOME/runs/AnJuXiaoBaoKWS"
```

每次新开 SSH 终端后，先执行：

```bash
conda activate cosyvoice310
source ~/.anju_xiaobao_kws_env
```

## 纯净 TTS 生成任务

正式建议先在服务器只生成 pure wake-word clean TTS，回传本机后再做 RK3566 底噪混合和远场模拟。

```bash
tmux new -s clean_tts
conda activate cosyvoice310
source ~/.anju_xiaobao_kws_env
cd "$ANJU_PROJECT"
bash scripts/server_run_clean_cosyvoice3.sh
```

脚本会自动生成：

```text
~/runs/AnJuXiaoBaoKWS/<run_name>/
├── command.sh
├── stdout.log
├── stderr.log
└── run_meta.json
```

默认输出正式 clean 数据集路径为：

```bash
~/datasets/AnJuXiaoBaoKWS/data/anju_xiaobao_cosyvoice3_clean_purewake_YYYYMMDD_HHMMSS
```

clean 数据集包含：

```text
clean_positive/
clean_negative/
prompt_voices/
manifests/all_clean.jsonl
manifests/clean_positive.jsonl
manifests/clean_negative.jsonl
manifests/selected_prompts.jsonl
summary.json
```

正式脚本的默认策略：

```text
AISHELL3 + AISHELL 全量扫描
男女 speaker 尽量 1:1
每个 speaker 生成 6 条纯“安居小宝”正样本
每个 speaker 生成 2 条 hard negative
不启用同音负样本
不在服务器做远场/底噪混合
```

## 远场生成任务

如果确实需要服务器端一并生成远场混合数据，可以运行旧的 farfield 包装脚本：

```bash
tmux new -s farfield_tts
conda activate cosyvoice310
source ~/.anju_xiaobao_kws_env
cd "$ANJU_PROJECT"
bash scripts/server_run_farfield_cosyvoice3.sh
```

脚本会自动生成：

```text
~/runs/AnJuXiaoBaoKWS/<run_name>/
├── command.sh
├── stdout.log
├── stderr.log
└── run_meta.json
```

默认输出数据集路径为：

```bash
~/datasets/AnJuXiaoBaoKWS/data/anju_xiaobao_farfield_cosyvoice3_YYYYMMDD_HHMMSS
```

默认不覆盖已有数据集，避免误删正式数据。如果确实要复用固定目录，先手工确认目标目录内容，再清理或指定新的输出目录。

## 检查命令

检查目录：

```bash
ls ~/projects ~/models ~/datasets ~/outputs ~/runs
```

检查空间占用：

```bash
du -sh ~/datasets/AnJuXiaoBaoKWS ~/models
```

查看任务日志：

```bash
tail -n 50 ~/runs/AnJuXiaoBaoKWS/<run_name>/stdout.log
tail -n 50 ~/runs/AnJuXiaoBaoKWS/<run_name>/stderr.log
```

## 清理 Smoke/Dryrun 数据

CosyVoice3 的 smoke/dryrun 测试数据可以用专用脚本清理。默认只预览，不删除：

```bash
conda activate cosyvoice310
source ~/.anju_xiaobao_kws_env
cd "$ANJU_PROJECT"

bash scripts/server_cleanup_cosyvoice3_smoke.sh
```

确认列表只包含临时测试目录后，再执行删除：

```bash
bash scripts/server_cleanup_cosyvoice3_smoke.sh --apply
```

如果需要非交互清理：

```bash
bash scripts/server_cleanup_cosyvoice3_smoke.sh --apply --yes
```

该脚本只匹配 `smoke` 和 `dryrun` 命名的 CosyVoice3 临时目录，不会清理基础数据集、正式 clean 数据集或模型目录。

## 使用原则

- `projects/` 只放源码。
- `models/` 只放可复用模型权重。
- `datasets/` 只放正式训练和评估数据集。
- `outputs/` 放可清理的试听和调试输出。
- `runs/` 保存每次长任务的命令和日志，便于复现。
- `downloads/` 只是下载中转目录。
- `tmp/` 可随时清理。
- `archive/` 只放旧版本备份，不作为默认训练输入。
