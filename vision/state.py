from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision.types import Detection


class SharedFrameState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame_jpeg: Optional[bytes] = None
        self._detections: List[Detection] = []
        self._status = "starting"
        self._frame_index = 0
        self._fps = 0.0

    def update(
        self,
        annotated_frame: Optional[np.ndarray],
        detections: List[Detection],
        status: str,
        fps: float,
    ) -> None:
        with self._condition:
            self._detections = list(detections)
            self._status = status
            self._fps = fps

            if annotated_frame is not None:
                success, encoded = cv2.imencode(".jpg", annotated_frame)
                self._frame_jpeg = encoded.tobytes() if success else None
                self._frame_index += 1
            self._condition.notify_all()

    def wait_for_frame(self, last_index: int, timeout: float = 1.0) -> Tuple[int, Optional[bytes]]:
        with self._condition:
            if self._frame_index <= last_index:
                self._condition.wait(timeout=timeout)
            return self._frame_index, self._frame_jpeg

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "fps": self._fps,
                "frame_index": self._frame_index,
                "detections": [
                    {
                        "label": d.label,
                        "confidence": round(d.confidence, 3),
                        "bbox": list(d.bbox),
                    }
                    for d in self._detections
                ],
            }
