from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from vision.types import Detection

LOGGER = logging.getLogger("astra_vision")


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


class HorizonYoloBinDetector:
    """Use the provided Horizon .bin YOLO model for inference."""

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        input_size: int = 640,
        class_names: List[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size
        self.class_names = class_names or ["person"]
        self._model = None
        self._logged_output_layout = False

        try:
            import hobot_dnn.pyeasy_dnn as dnn  # type: ignore

            self._dnn = dnn
            models = dnn.load(model_path)
            self._model = models[0]
            LOGGER.info("Loaded YOLO model: %s", model_path)
        except Exception as exc:
            self._dnn = None
            self._model = None
            LOGGER.error("Failed to load Horizon model %s: %s", model_path, exc)

    @staticmethod
    def _letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, scale, left, top

    @staticmethod
    def _bgr_to_nv12(img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        yuv_i420 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        y_size = h * w
        uv_h = h // 2
        u = yuv_i420[y_size : y_size + y_size // 4].reshape((uv_h, w // 2))
        v = yuv_i420[y_size + y_size // 4 :].reshape((uv_h, w // 2))
        uv = np.empty((uv_h, w), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v
        return np.concatenate((yuv_i420[:y_size], uv.reshape(-1)))

    def _forward(self, nv12_input: np.ndarray):
        if self._model is None:
            return []

        forward_variants = [
            lambda: self._model.forward(nv12_input),
            lambda: self._model.forward([nv12_input]),
            lambda: self._model.forward(nv12_input.tobytes()),
        ]
        last_exc = None
        for run in forward_variants:
            try:
                out = run()
                return out if isinstance(out, list) else [out]
            except Exception as exc:
                last_exc = exc
        LOGGER.error("Model forward failed: %s", last_exc)
        return []

    @staticmethod
    def _to_numpy(output_obj) -> np.ndarray | None:
        if output_obj is None:
            return None
        if isinstance(output_obj, np.ndarray):
            return output_obj
        if hasattr(output_obj, "buffer"):
            try:
                arr = np.array(output_obj.buffer)
                if arr.dtype == object and arr.size == 1 and isinstance(arr.item(), (bytes, bytearray, memoryview)):
                    raw = np.frombuffer(arr.item(), dtype=np.float32)
                    return raw
                return arr
            except Exception:
                return None
        return None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _reshape_to_matrix(arr: np.ndarray) -> np.ndarray | None:
        # 统一拉平为 [N, C] 形态，方便后处理
        if arr.ndim == 1:
            if arr.size % 6 == 0:
                return arr.reshape(-1, 6)
            return None

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
                if arr.ndim == 2:
                    # 兼容 [C, N] -> [N, C]
                    if arr.shape[0] in (6, 7, 84, 85):
                        return arr.transpose(1, 0)
                    return arr
            if arr.shape[-1] in (6, 7, 84, 85):
                return arr.reshape(-1, arr.shape[-1])
            if arr.shape[0] in (6, 7, 84, 85):
                return arr.transpose(1, 2, 0).reshape(-1, arr.shape[0])
            return None

        if arr.ndim == 4:
            # 常见 [1, C, H, W] / [1, H, W, C]
            if arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3:
                return None
            if arr.shape[0] in (6, 7, 84, 85):
                return arr.transpose(1, 2, 0).reshape(-1, arr.shape[0])
            if arr.shape[-1] in (6, 7, 84, 85):
                return arr.reshape(-1, arr.shape[-1])
            return None

        return None

    def _decode_matrix(
        self,
        matrix: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        src_w: int,
        src_h: int,
    ) -> List[Detection]:
        if matrix.shape[1] < 6:
            return []

        # 案例A: [x1,y1,x2,y2,score,cls]
        if matrix.shape[1] < 20:
            boxes = matrix[:, :4].astype(np.float32)
            scores = matrix[:, 4].astype(np.float32)
            cls_ids = matrix[:, 5].astype(np.int32)

            # score 在 logit 空间时自动过 sigmoid
            if np.nanmax(scores) > 1.5 or np.nanmin(scores) < -0.5:
                scores = self._sigmoid(scores)

            # 若坐标明显在 0~1，认为是归一化坐标
            if np.nanmax(boxes) <= 1.5:
                boxes[:, [0, 2]] *= float(self.input_size)
                boxes[:, [1, 3]] *= float(self.input_size)

            # 若是 xywh，需要转 xyxy
            if np.mean(boxes[:, 2] >= boxes[:, 0]) < 0.6 or np.mean(boxes[:, 3] >= boxes[:, 1]) < 0.6:
                boxes = _xywh2xyxy(boxes)

        else:
            # 案例B: [x,y,w,h,obj,cls...] 或 [x1,y1,x2,y2,obj,cls...]
            boxes_raw = matrix[:, :4].astype(np.float32)
            obj = matrix[:, 4].astype(np.float32)
            cls_scores = matrix[:, 5:].astype(np.float32)

            if np.nanmax(obj) > 1.5 or np.nanmin(obj) < -0.5:
                obj = self._sigmoid(obj)
            if np.nanmax(cls_scores) > 1.5 or np.nanmin(cls_scores) < -0.5:
                cls_scores = self._sigmoid(cls_scores)

            cls_ids = cls_scores.argmax(axis=1).astype(np.int32)
            cls_conf = cls_scores.max(axis=1)
            scores = obj * cls_conf

            # 判断 xyxy/xywh
            is_xyxy = np.mean(boxes_raw[:, 2] > boxes_raw[:, 0]) > 0.8 and np.mean(boxes_raw[:, 3] > boxes_raw[:, 1]) > 0.8
            boxes = boxes_raw if is_xyxy else _xywh2xyxy(boxes_raw)

            if np.nanmax(boxes) <= 1.5:
                boxes[:, [0, 2]] *= float(self.input_size)
                boxes[:, [1, 3]] *= float(self.input_size)

        valid = scores >= self.conf_thres
        boxes = boxes[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]
        if boxes.shape[0] == 0:
            return []

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-6)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, src_w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, src_h - 1)

        keep = _nms(boxes, scores, self.iou_thres)
        detections: List[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            cls_id = int(cls_ids[i])
            label = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"cls_{cls_id}"
            detections.append(
                Detection(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    def _decode(self, outputs, scale: float, pad_x: int, pad_y: int, src_w: int, src_h: int) -> List[Detection]:
        candidates: List[Detection] = []

        for out in outputs:
            arr = self._to_numpy(out)
            if arr is None or arr.size == 0:
                continue

            if not self._logged_output_layout:
                LOGGER.info("YOLO output tensor shape=%s dtype=%s", tuple(arr.shape), arr.dtype)

            matrix = self._reshape_to_matrix(arr)
            if matrix is None or matrix.size == 0:
                continue

            if not self._logged_output_layout:
                LOGGER.info("YOLO reshaped matrix=%s", tuple(matrix.shape))
                self._logged_output_layout = True

            candidates.extend(self._decode_matrix(matrix, scale, pad_x, pad_y, src_w, src_h))

        return candidates

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0 or self._model is None:
            return []

        src_h, src_w = frame_bgr.shape[:2]
        padded, scale, pad_x, pad_y = self._letterbox(frame_bgr, self.input_size)
        nv12 = self._bgr_to_nv12(padded)
        outputs = self._forward(nv12)
        return self._decode(outputs, scale, pad_x, pad_y, src_w, src_h)
