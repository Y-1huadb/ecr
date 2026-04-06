# Astra Camera Vision Service

## 新结构

- 根入口：`main.py`
- YOLO 模型调用：`vision/yolo/detector.py`
- Web 服务：`vision/web/server.py`
- 运行编排：`vision/runtime/service.py`
- 兼容入口：`astraCamera/vision_app.py`

## 模型

默认使用你提供的模型：

- `models/yolov12n_detect_bayese_640x640_nv12_modified.bin`

## 启动

在项目根目录运行：

```bash
python3 main.py
```

默认网页地址：`http://0.0.0.0:8080`

## 可选参数

- `--host`：网页监听地址
- `--port`：网页端口
- `--model`：YOLO `.bin` 模型路径
- `--topic`：ROS2 topic 名称
- `--fps`：采集和处理帧率
- `--confidence`：检测阈值

示例：

```bash
python3 main.py --model models/yolov12n_detect_bayese_640x640_nv12_modified.bin --host 0.0.0.0 --port 8080 --topic /astra/color/image/compressed --fps 30 --confidence 0.35
```
