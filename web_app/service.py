import argparse
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

from action_detection.yolo26_pose import Ultralytics_YOLO_Pose_Bayese_YUV420SP
from astra_camera.astra_camera import Astra_Camera
from object_detection.yolo26_det import (
    Ultralytics_YOLO_Detect_Bayese_YUV420SP,
    coco_names,
    draw_detection,
)


@dataclass
class RuntimeConfig:
    det_model_path: str
    pose_model_path: str
    host: str
    port: int
    jpeg_quality: int
    score_thres: float
    nms_thres: float
    pose_kpt_conf_thres: float


class InferenceRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.camera = Astra_Camera()
        self.detector = None
        self.pose_detector = None

        self._latest_jpeg: Optional[bytes] = None
        self._latest_meta: Dict[str, Any] = {
            "timestamp": 0,
            "fps": 0.0,
            "detections": [],
            "poses": [],
        }

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def initialize(self) -> None:
        if not self.camera.initialize():
            raise RuntimeError("AstraCamera 初始化失败。")

        self.detector = Ultralytics_YOLO_Detect_Bayese_YUV420SP(
            model_path=self.config.det_model_path,
            classes_num=80,
            nms_thres=self.config.nms_thres,
            score_thres=self.config.score_thres,
            reg=16,
            strides=[8, 16, 32],
        )

        self.pose_detector = Ultralytics_YOLO_Pose_Bayese_YUV420SP(
            model_path=self.config.pose_model_path,
            classes_num=1,
            nms_thres=self.config.nms_thres,
            score_thres=self.config.score_thres,
            reg=16,
            strides=[8, 16, 32],
            nkpt=17,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self.camera.release()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_meta(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._latest_meta))

    def _run_detector(self, frame_bgr) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
        input_tensor = self.detector.preprocess_yuv420sp(frame_bgr)
        outputs = self.detector.c2numpy(self.detector.forward(input_tensor))
        results = self.detector.postProcess(outputs)

        payload = []
        for class_id, score, x1, y1, x2, y2 in results:
            payload.append(
                {
                    "label": coco_names[class_id],
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return payload, results

    def _run_pose(self, frame_bgr) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
        input_tensor = self.pose_detector.preprocess_yuv420sp(frame_bgr)
        outputs = self.pose_detector.c2numpy(self.pose_detector.forward(input_tensor))
        results = self.pose_detector.postProcess(outputs)

        payload = []
        for class_id, score, x1, y1, x2, y2, kpts in results:
            payload.append(
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "keypoints": [
                        [int(x), int(y), float(conf)] for x, y, conf in kpts
                    ],
                }
            )
        return payload, results

    def run_loop(self) -> None:
        last_time = time.time()
        while not self._stop_event.is_set():
            frame_bgr, _ = self.camera.read_frames()
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            vis = frame_bgr.copy()

            det_payload, det_results = self._run_detector(frame_bgr)
            for class_id, score, x1, y1, x2, y2 in det_results:
                draw_detection(vis, (x1, y1, x2, y2), score, class_id)

            pose_payload, pose_results = self._run_pose(frame_bgr)
            kpt_conf_inverse = -np.log(1 / self.config.pose_kpt_conf_thres - 1)
            for _, _, _, _, _, _, kpts in pose_results:
                for idx, (x, y, conf) in enumerate(kpts):
                    if conf < kpt_conf_inverse:
                        continue
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        str(idx),
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            cv2.putText(
                vis,
                f"FPS: {fps:.2f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, encoded = cv2.imencode(
                ".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
            )
            if not ok:
                continue

            with self._lock:
                self._latest_jpeg = encoded.tobytes()
                self._latest_meta = {
                    "timestamp": int(now * 1000),
                    "fps": round(fps, 2),
                    "detections": det_payload,
                    "poses": pose_payload,
                }


def create_app(runtime: InferenceRuntime, web_dir: str) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return send_from_directory(web_dir, "index.html")

    @app.route("/video_feed")
    def video_feed():
        def stream():
            while True:
                frame = runtime.latest_jpeg()
                if frame is None:
                    time.sleep(0.02)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/latest")
    def api_latest():
        return jsonify(runtime.latest_meta())

    return app


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="AstraCamera + YOLO26(det/pose) 实时网页展示"
    )
    parser.add_argument(
        "--det-model-path",
        type=str,
        default="/home/sunrise/Desktop/ECR/models/yolov12n_detect_bayese_640x640_nv12_modified.bin",
        help="目标检测模型路径",
    )
    parser.add_argument(
        "--pose-model-path",
        type=str,
        default="/home/sunrise/Desktop/ECR/models/yolo11n_pose_bayese_640x640_nv12.bin",
        help="姿态估计模型路径",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--score-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.7)
    parser.add_argument("--pose-kpt-conf-thres", type=float, default=0.5)

    args = parser.parse_args()
    return RuntimeConfig(
        det_model_path=args.det_model_path,
        pose_model_path=args.pose_model_path,
        host=args.host,
        port=args.port,
        jpeg_quality=args.jpeg_quality,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        pose_kpt_conf_thres=args.pose_kpt_conf_thres,
    )


def main() -> None:
    config = parse_args()
    runtime = InferenceRuntime(config)
    runtime.initialize()

    worker = threading.Thread(target=runtime.run_loop, daemon=True)
    worker.start()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    web_dir = os.path.join(project_root, "web_ui")
    app = create_app(runtime, web_dir=web_dir)
    try:
        app.run(host=config.host, port=config.port, threaded=True)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
