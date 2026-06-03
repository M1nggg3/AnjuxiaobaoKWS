# M1 物理远场训练数据自动采集

M1 通过笔记本扬声器播放 clean TTS，并由 RK3566 的真实麦克风录制 raw-only 训练数据。本机使用 Wi-Fi ADB 连接 `192.168.3.228:44891` 逐条控制录制、拉取与校验。

## 准备

确认无线 ADB 在线：

```powershell
adb connect 192.168.3.228:44891
adb -s 192.168.3.228:44891 get-state
```

生成固定的 100 条正样本和 50 条近音负样本计划：

```powershell
$env:PYTHONPATH = "E:\CodeWorking\Project\AnJuXiaoBaoKWS\src"
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.prepare_m1_capture_plan
```

程序输出的 `output_dir` 是后续命令中的 `<M1_DIR>`。

## 采集

先摆放为 1m 并进行冒烟验证：

```powershell
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.run_m1_capture `
  --experiment-dir "<M1_DIR>" `
  --distance 1m `
  --adb-serial 192.168.3.228:44891 `
  --smoke
```

人工抽听冒烟结果中的 `raw.wav`，确认无系统提示音、唤醒反馈音、截断或严重削波后，再执行正式三档采集。

冒烟音频单独保存到 `qc/smoke_captures`，不会被写入正式训练 manifest。即使冒烟条件无需调整，正式 `1m` 批次仍会重新录制对应源样本。

```powershell
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.run_m1_capture --experiment-dir "<M1_DIR>" --distance 1m --adb-serial 192.168.3.228:44891 --resume
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.run_m1_capture --experiment-dir "<M1_DIR>" --distance 3m --adb-serial 192.168.3.228:44891 --resume
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.run_m1_capture --experiment-dir "<M1_DIR>" --distance 5m --adb-serial 192.168.3.228:44891 --resume
```

不传 `--playback-device` 时，Windows 当前默认输出设备用于外放。全量采集开始后不要调整系统音量、输出设备、设备朝向或板端配置。

## 汇总

```powershell
D:\conda-envs\cosyvoice310\python.exe -m anju_kws.collection.report_m1_capture `
  --experiment-dir "<M1_DIR>"
```

训练只使用 `manifests/training_raw_positive.jsonl` 和 `manifests/training_raw_negative.jsonl` 中登记的校验通过样本。日志与状态文件保留用于后续诊断。
