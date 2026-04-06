from __future__ import annotations

import logging
import threading
import time
from typing import List

import cv2
import numpy as np

from astraCamera.astraCamera import AstraCamera
from vision.publishers.ros2_publisher import RosCompressedImagePublisher
from vision.state import SharedFrameState
from vision.types import Detection
from vision.yolo.detector import HorizonYoloBinDetector

LOGGER = logging.getLogger("astra_vision")


class AstraVisionService:
    def __init__(
        self,
        model_path: str,
        topic_name: str,
        fps: int,
        min_confidence: float,
    ) -> None:
        self.topic_name = topic_name
        self.target_fps = max(fps, 1)
        self.min_confidence = min_confidence

        self.camera = AstraCamera()
        self.detector = HorizonYoloBinDetector(model_path=model_path, conf_thres=min_confidence)
        self.publisher = RosCompressedImagePublisher(topic_name=topic_name)
        self.state = SharedFrameState()

        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._loop, daemon=True, name="vision-worker")

    def start(self) -> None:
        if not self.camera.initialize():
            raise RuntimeError("Astra camera initialization failed")
        self.worker.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        self.publisher.close()
        self.camera.release()

    @staticmethod
    def _draw(frame_bgr: np.ndarray, detections: List[Detection], fps: float, topic_name: str, min_conf: float) -> np.ndarray:
        canvas = frame_bgr.copy()
        overlay = canvas.copy()
        h, w = canvas.shape[:2]

        for det in detections:
            if det.confidence < min_conf:
                continue
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            color = (0, 215, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - th - baseline - 6)
            cv2.rectangle(overlay, (x1, top), (x1 + tw + 12, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 6, y1 - baseline - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
        bar_h = 54
        panel = canvas.copy()
        cv2.rectangle(panel, (0, 0), (w, bar_h), (16, 18, 31), -1)
        cv2.addWeighted(panel, 0.62, canvas, 0.38, 0, canvas)
        cv2.putText(canvas, f"Astra YOLO | FPS {fps:.1f} | Topic {topic_name}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def _loop(self) -> None:
        period = 1.0 / float(self.target_fps)
        prev = time.time()
        fps = 0.0

        while not self.stop_event.is_set():
            tic = time.time()
            color_frame, _ = self.camera.read_frames()
            if color_frame is None:
                self.state.update(None, [], "waiting for color stream", fps)
                time.sleep(0.05)
                continue

            detections = self.detector.detect(color_frame)
            annotated = self._draw(color_frame, detections, fps, self.topic_name, self.min_confidence)
            self.publisher.publish(color_frame)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            status = f"streaming {len(detections)} detections"
            self.state.update(annotated, detections, status, fps)

            spend = time.time() - tic
            if spend < period:
                time.sleep(period - spend)
