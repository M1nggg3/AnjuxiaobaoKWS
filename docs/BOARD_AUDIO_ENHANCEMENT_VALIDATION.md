# RK3566 板端音频增强验证流程

当前 Android 端已经增加两路音频保存：

```text
raw PCM：原始 AudioRecord 输入
enhanced PCM：DC remove + 80Hz high-pass + AGC + limiter 后送入模型的输入
```

板端路径：

```text
/sdcard/Android/data/cn.org.wenet.wekws/files/logs/
/sdcard/Android/data/cn.org.wenet.wekws/files/captures/
```

本机拉取脚本：

```powershell
E:\CodeWorking\Project\AnJuXiaoBaoKWS\scripts\pull_latest_android_session.ps1
```

## 验证方法

分别做三轮监听，每轮只录一种距离：

```text
1. 近距离：约 20 cm
2. 中距离：约 1 m
3. 远距离：约 2 m
```

每轮操作：

```text
1. 打开 App，点击开始监听
2. 用固定音量说 5 到 10 次“安居小宝”
3. 点击停止监听
4. 运行 pull_latest_android_session.ps1 拉取日志和 raw/enhanced 音频
```

## 重点观察字段

session log 中会出现：

```text
audio_debug read=...
raw_rms=...
raw_peak=...
enhanced_rms=...
enhanced_peak=...
gain_db=...
clipped=...
noise_floor=...
speech=true/false
score=...
native_debug ...
```

判断标准：

```text
如果 raw_rms 很低但 enhanced_rms 明显提升，说明板端增强有效。
如果 enhanced_rms 提升后 score 仍不上升，说明主要问题转向训练数据/模型泛化。
如果 clipped 很高，说明增益过大，需要降低 AGC 目标或最大增益。
如果 speech=false 但实际在说话，说明动态语音阈值仍偏高。
```
