# Android 安装包

当前 APK：

```text
anjuxiaobao-kws-debug-sliding-beep-20260603.apk
```

说明：

- 基于 WeKWS Android runtime。
- 支持原模型和当前 M2/M1 远场模型切换。
- 当前实时检测采用 1.2s 滑动窗口 + CTC prefix beam search。
- 唤醒成功后播放单次短提示音。

安装命令：

```powershell
adb install -r deploy\android_app\anjuxiaobao-kws-debug-sliding-beep-20260603.apk
```
