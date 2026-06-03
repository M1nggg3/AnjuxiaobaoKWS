# 安居小宝语音唤醒项目

本项目用于在 RK3566 板端实现“安居小宝”语音唤醒。当前可用版本基于 WeKWS FSMN-CTC 模型，Android 端采用 raw PCM 输入、滑动窗口打分、CTC prefix beam search 和多模型切换机制。

项目已经打通以下链路：

- TTS/远场采集数据准备
- WeKWS 模型训练与微调
- checkpoint 转 ONNX
- Android App 集成 ONNX Runtime
- RK3566 实时录音推理
- 板端日志、raw PCM 录音保存
- 原模型与新模型 A/B 切换测试

## 当前状态

当前 Android App 已可在 RK3566 上运行，支持多个模型切换。最新可用的板端逻辑为：

```text
AudioRecord raw PCM
→ 1.2s PCM ring buffer
→ 每 100ms 取最近窗口
→ native scoreWindow
→ 80-bin fbank
→ context expansion 到 400 维
→ ONNX 推理
→ CTC prefix beam search
→ 连续 2 次命中后触发
→ 触发后播放单次短提示音
```

当前 APK 已单独放在：

```text
deploy/android_app/anjuxiaobao-kws-debug-sliding-beep-20260603.apk
```

安装命令：

```powershell
adb install -r deploy\android_app\anjuxiaobao-kws-debug-sliding-beep-20260603.apk
```

指定设备安装：

```powershell
adb -s <adb_serial> install -r deploy\android_app\anjuxiaobao-kws-debug-sliding-beep-20260603.apk
```

## 仓库目录

```text
configs/                    # 数据、训练配置
data/                       # 训练清单、manifest、少量可公开的元数据，不存放真实音频
deploy/                     # 部署产物，包括 Android APK 和历史模型包
dict/                       # WeKWS 字典、lexicon
docs/                       # 项目文档和阶段方案
models/                     # 小体积模型产物、ONNX、checkpoint
scripts/                    # 本机和服务器自动化脚本
src/anju_kws/               # 数据处理、TTS、采集、评估、部署脚本
tests/                      # 本机脚本测试
third_party/wekws/          # WeKWS 源码和 Android runtime
tools/                      # 辅助工具
_archive/                   # 已归档旧版本，不作为默认输入
```

## Android 工程

Android 工程位置：

```text
third_party/wekws/runtime/android
```

主要文件：

```text
third_party/wekws/runtime/android/app/src/main/java/cn/org/wenet/wekws/MainActivity.java
third_party/wekws/runtime/android/app/src/main/java/cn/org/wenet/wekws/Spot.java
third_party/wekws/runtime/android/app/src/main/java/cn/org/wenet/wekws/FarfieldCaptureService.java
third_party/wekws/runtime/android/app/src/main/cpp/wekws.cc
third_party/wekws/runtime/android/app/src/main/assets/model_registry.json
third_party/wekws/runtime/android/app/src/main/assets/models/
```

重新编译：

```powershell
cd third_party\wekws\runtime\android
.\gradlew.bat :app:assembleDebug
```

编译后的 APK：

```text
third_party/wekws/runtime/android/app/build/outputs/apk/debug/app-debug.apk
```

## 内置模型

Android App 通过 `model_registry.json` 管理多个模型。当前已集成：

```text
original_mid_20260509
m1_rawonly_finetune
m1_rawonly_pretrain_full
m2_m1_neg2500_pretrained_finetune
```

模型 assets 目录：

```text
third_party/wekws/runtime/android/app/src/main/assets/models/
```

每个模型目录通常包含：

```text
kws.onnx
kws_runtime_config.json
```

运行配置中的关键参数：

```text
threshold_initial
speech_rms_threshold
speech_peak_threshold
sliding_window_ms
sliding_hop_ms
sliding_consecutive_hits
sliding_cooldown_ms
enable_wakeup_tone
```

说明：

- `original_mid_20260509` 是远场专项之前的原模型，适合做 A/B 对比。
- `m2_m1_neg2500_pretrained_finetune` 是后续结合 M1/M2 远场数据和负样本训练后的 WeKWS 模型。
- 当前 App 默认模型以 `model_registry.json` 中 `default_model_id` 为准。

## WeKWS 训练结果与部署产物

当前仓库内保留了部分可复用的小体积模型产物：

```text
models/wekws/fsmn_ctc_farfield_main_20260521_001/
deploy/rk3566_wekws_model_mid_20260509/
```

其中典型文件包括：

```text
*.pt                 # PyTorch checkpoint
*.yaml               # 对应训练配置或 checkpoint 元信息
kws.onnx             # Android / ONNX Runtime 推理模型
kws_runtime_config.json
config.yaml
dict.txt
words.txt
```

ONNX 导出脚本：

```text
src/anju_kws/deploy/export_fsmn_ctc_onnx.py
```

## 板端日志与录音

Android App 监听时会保存日志和 raw PCM：

```text
/sdcard/Android/data/cn.org.wenet.wekws/files/logs/listen_session_*.log
/sdcard/Android/data/cn.org.wenet.wekws/files/captures/listen_session_*_raw_16k_s16le.pcm
```

本机拉取脚本：

```text
scripts/pull_latest_android_session.ps1
```

日志中建议重点查看：

```text
session_start
audio_debug
sliding_window
native_debug window_decode_debug
wakeup_tone_played
session_end
```

## 数据采集脚本

本项目实现过两类板端采集：

### M1 单板采集

```text
src/anju_kws/collection/prepare_m1_capture_plan.py
src/anju_kws/collection/run_m1_capture.py
src/anju_kws/collection/report_m1_capture.py
```

### M2 三板同步采集

用于三台 RK3566 同步录制 1m / 3m / 5m 远场数据：

```text
src/anju_kws/collection/prepare_m2_parallel_capture_plan.py
src/anju_kws/collection/run_m2_parallel_capture.py
src/anju_kws/collection/report_m2_parallel_capture.py
```

典型流程：

```text
生成播放/录制计划
→ 三台板端通过 Wi-Fi ADB 连接
→ 本机播放 TTS 音源
→ 三台 RK3566 同步录制 raw PCM
→ 本机拉回录音
→ 生成 recordings.jsonl 和训练 manifest
```

## 离线评估脚本

常用评估脚本：

```text
src/anju_kws/eval/replay_android_streaming.py
src/anju_kws/eval/sliding_window_score_ctc_eval.py
src/anju_kws/eval/sliding_window_realvoice_eval.py
src/anju_kws/eval/run_board_ab_playback_test.py
scripts/eval_board_raw_playback_segments.py
```

评估建议分三层：

```text
1. 离线切片 score_ctc
2. 离线滑窗 replay
3. RK3566 板端实时监听
```

这样可以区分模型能力、流式逻辑和板端实时输入之间的差异。

## 服务器脚本

服务器相关脚本主要用于 CosyVoice3 clean TTS 生成、远场混合、WeKWS 训练和评估：

```text
scripts/server_setup_anju_workspace.sh
scripts/server_run_clean_cosyvoice3.sh
scripts/server_run_mix_cosyvoice3_farfield.sh
scripts/server_run_wekws_farfield_train.sh
scripts/server_run_wekws_farfield_mined_train.sh
scripts/server_eval_wekws_score_ctc.sh
scripts/server_watch_mix_progress.sh
scripts/server_watch_wekws_train.sh
```

服务器个人目录约定见：

```text
docs/SERVER_WORKSPACE_LAYOUT.md
```

## 数据与隐私说明

本仓库不应该上传真实录音、板端抓取音频、TTS 生成音频或完整大型数据集。

`.gitignore` 已排除：

```text
*.wav
*.pcm
*.mp3
*.flac
*.m4a
*.zip
*.tar
*.tar.gz
*.tgz
*.7z
*.rar
**/build/
**/.gradle/
**/.cxx/
local.properties
```

真实数据集目前应保存在本机或服务器的数据目录，例如：

```text
E:\CodeWorking\datasets\AnJuXiaoBaoKWS\
/home/clm/datasets/AnJuXiaoBaoKWS/
```

## 当前经验结论

- 滑动窗口板端检测明显优于长连续流检测，尤其适合远场唤醒。
- 当前 1.2s 窗口、100ms hop、连续 2 次命中的策略已经可以在板端稳定触发。
- 阈值不能只按近距离调，3m / 5m / 7m 的 score 较低，直接提高阈值会显著损伤远场召回。
- 误唤醒率需要单独用办公室连续负样本测试，不能只看正样本唤醒测试。
- 如果要优化误唤醒，优先考虑二级触发策略，例如 hit cluster 时长、cluster 最高分、冷却时间，而不是简单提高阈值。

## 参考文档

```text
docs/CURRENT_PROJECT_ARCHITECTURE_20260513.md
docs/M1_PHYSICAL_FARFIELD_COLLECTION.md
docs/BOARD_AUDIO_ENHANCEMENT_VALIDATION.md
docs/SERVER_WORKSPACE_LAYOUT.md
deploy/android_app/README.md
```

## 快速接手建议

1. 先安装 `deploy/android_app/anjuxiaobao-kws-debug-sliding-beep-20260603.apk` 到 RK3566。
2. 在 App 中确认当前选择的模型。
3. 做 1m / 3m / 5m / 7m 正样本唤醒测试。
4. 拉取 `listen_session_*.log` 和 raw PCM。
5. 用离线 replay 脚本复核板端实时结果。
6. 再做 10 到 20 分钟纯负样本连续测试，统计误唤醒率。

