# ECR Astra + YOLO26 Web Demo

该示例将 `astraCamera` 采集到的图像交给 `yolo26_det` 与 `yolo26_pose` 推理，并通过网页实时展示。

## 目录

- `main.py`: 程序入口（包含 `main` 调用）
- `web_app/service.py`: 后端服务与推理循环
- `web_ui/index.html`: 前端页面（独立文件夹）

## 运行前准备

1. 确保在 RDK 设备上可用 `AstraSDK` 与相机。
2. 准备目标检测与姿态模型文件。
3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py \
  --det-model-path /path/to/det_model.bin \
  --pose-model-path /path/to/pose_model.bin \
  --host 0.0.0.0 \
  --port 8080
```

浏览器访问：

```text
http://<你的设备IP>:8080
```

## 可选参数

- `--score-thres`: 置信度阈值（默认 `0.25`）
- `--nms-thres`: NMS 阈值（默认 `0.7`）
- `--pose-kpt-conf-thres`: 关键点阈值（默认 `0.5`）
- `--jpeg-quality`: 视频流 JPEG 质量（默认 `85`）
