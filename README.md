# 安居小宝语音唤醒项目

当前版本：WeKWS FSMN-CTC RK3566 Android 实时唤醒版本。

详细架构说明见：

`E:\CodeWorking\Project\AnJuXiaoBaoKWS\docs\CURRENT_PROJECT_ARCHITECTURE_20260513.md`

## 当前保留版本

```text
实验目录：experiments\pretrain_posbalanced_mid_20260509_001
训练数据：data\prepared_pretrain_posbalanced_mid_20260509
字典目录：dict\pretrain_posbalanced_mid_20260509
部署包：deploy\rk3566_wekws_model_mid_20260509
Android 工程：third_party\wekws\runtime\android
```

当前部署模型来自：

`experiments\pretrain_posbalanced_mid_20260509_001\5.pt`

当前 Android 端使用：

```text
third_party\wekws\runtime\android\app\src\main\assets\kws.onnx
third_party\wekws\runtime\android\app\src\main\assets\kws_runtime_config.json
```

旧版本已归档到：

`E:\CodeWorking\Project\AnJuXiaoBaoKWS\_archive\superseded_20260513`

## 数据说明

仓库上传代码、配置、模型文件、训练清单和部署资源。真实录音、板端抓取音频、生成音频等声音数据默认不进入 Git 仓库，避免公开真实声纹和本地采集数据。
