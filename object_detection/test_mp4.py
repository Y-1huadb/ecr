#!/usr/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
# Modified to support both image input and MP4 input/output.

import os
import cv2
import numpy as np
from pathlib import Path

# scipy
try:
    from scipy.special import softmax
except Exception:
    print("scipy is not installed, installing.")
    os.system("pip install scipy")
    from scipy.special import softmax

# hobot_dnn
try:
    try:
        from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
    except Exception:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn  # BSP Python API from PyPI
except Exception:
    print("pip install hobot-dnn-rdkx5")
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

from time import time
import argparse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RDK_YOLO")

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/sunrise/Desktop/ECR/models/yolov12n_detect_bayese_640x640_nv12_modified.bin',
        help='Path to BPU Quantized *.bin Model.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/home/sunrise/Desktop/mono2d/config/person_body.jpg',
        help='Input path. Supports image or video(mp4/avi/mov/mkv).'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='py_result.jpg',
        help='Output path. Image input -> image output; video input -> mp4 output.'
    )
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), default=[8, 16, 32], help='--strides 8,16,32')
    parser.add_argument('--save-fps', type=float, default=0.0, help='Override output video fps. 0 means use source fps.')
    parser.add_argument('--max-frames', type=int, default=-1, help='Only process first N frames for debugging. -1 means all.')
    opt = parser.parse_args()
    logger.info(opt)

    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. downloading ...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolov13n_detect_bayese_640x640_nv12.bin")
        opt.model_path = 'yolov13n_detect_bayese_640x640_nv12.bin'

    model = Ultralytics_YOLO_Detect_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        nms_thres=opt.nms_thres,
        score_thres=opt.score_thres,
        reg=opt.reg,
        strides=opt.strides
    )

    input_path = Path(opt.input)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTS:
        infer_image(model, opt.input, opt.output)
    elif suffix in VIDEO_EXTS:
        infer_video(model, opt.input, opt.output, save_fps=opt.save_fps, max_frames=opt.max_frames)
    else:
        raise ValueError(f"Unsupported input type: {opt.input}")


def infer_image(model, image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Load image failed: {image_path}")

    results = run_single_frame(model, img)

    logger.info("\033[1;32mDraw Results:\033[0m")
    for class_id, score, x1, y1, x2, y2 in results:
        logger.info("(%d, %d, %d, %d) -> %s: %.2f" % (x1, y1, x2, y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)

    cv2.imwrite(save_path, img)
    logger.info("\033[1;32m" + f'saved image in path: "{save_path}"' + "\033[0m")



def infer_video(model, video_path, save_path, save_fps=0.0, max_frames=-1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Open video failed: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0:
        src_fps = 25.0
    out_fps = save_fps if save_fps > 0 else src_fps

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # mp4v is usually the safest choice for OpenCV mp4 writing.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, out_fps, (src_w, src_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Open VideoWriter failed: {save_path}")

    logger.info(f"video info: {src_w}x{src_h}, fps={src_fps:.2f}, total_frames={total_frames}")
    logger.info(f"saving to: {save_path}, save_fps={out_fps:.2f}")

    frame_id = 0
    total_infer_ms = 0.0
    total_frames_done = 0
    whole_begin = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_id >= max_frames:
            break

        begin = time()
        results = run_single_frame(model, frame)
        infer_ms = (time() - begin) * 1000.0
        total_infer_ms += infer_ms
        total_frames_done += 1

        for class_id, score, x1, y1, x2, y2 in results:
            draw_detection(frame, (x1, y1, x2, y2), score, class_id)

        fps_text = 1000.0 / infer_ms if infer_ms > 0 else 0.0
        cv2.putText(
            frame,
            f"frame={frame_id} infer={infer_ms:.1f}ms fps={fps_text:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

        if frame_id % 10 == 0:
            logger.info(f"processed frame {frame_id}, infer={infer_ms:.2f} ms, dets={len(results)}")

        frame_id += 1

    cap.release()
    writer.release()

    whole_ms = (time() - whole_begin) * 1000.0
    avg_infer_ms = total_infer_ms / total_frames_done if total_frames_done > 0 else 0.0
    avg_infer_fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

    logger.info("\033[1;32m" + f"saved video in path: \"{save_path}\"" + "\033[0m")
    logger.info(f"processed frames: {total_frames_done}")
    logger.info(f"avg infer: {avg_infer_ms:.2f} ms/frame, avg infer fps: {avg_infer_fps:.2f}")
    logger.info(f"whole elapsed: {whole_ms:.2f} ms")



def run_single_frame(model, img):
    input_tensor = model.preprocess_yuv420sp(img)
    outputs = model.c2numpy(model.forward(input_tensor))
    return model.postProcess(outputs)


class Ultralytics_YOLO_Detect_Bayese_YUV420SP():
    def __init__(self, model_path, classes_num, nms_thres, score_thres, reg, strides):
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms" % (1000 * (time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s" % (model_path))
            logger.error("You can download the model file from the following docs: ./models/download.md")
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1 / self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        self.strides = strides
        logger.info(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f" % (self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f" % self.CONF_THRES_RAW)
        logger.info(f"{self.input_H = }, {self.input_W = }")
        logger.info(f"{self.strides = }")

        self.weights_static = np.array([i for i in range(reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        logger.info(f"{self.weights_static.shape = }")

        self.grids = []
        for stride in self.strides:
            assert self.input_H % stride == 0, f"{stride=}, {self.input_H=}: input_H % stride != 0"
            assert self.input_W % stride == 0, f"{stride=}, {self.input_W=}: input_W % stride != 0"
            grid_H, grid_W = self.input_H // stride, self.input_W // stride
            self.grids.append(np.stack([
                np.tile(np.linspace(0.5, grid_H - 0.5, grid_H), reps=grid_H),
                np.repeat(np.arange(0.5, grid_W + 0.5, 1), grid_W)
            ], axis=0).transpose(1, 0))
            logger.info(f"{self.grids[-1].shape = }")

    def preprocess_yuv420sp(self, img):
        RESIZE_TYPE = 0
        LETTERBOX_TYPE = 1
        PREPROCESS_TYPE = LETTERBOX_TYPE

        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
            self.y_scale = self.x_scale
            if self.x_scale <= 0 or self.y_scale <= 0:
                raise ValueError("Invalid scale factor.")

            new_w = int(self.img_w * self.x_scale)
            self.x_shift = (self.input_W - new_w) // 2
            x_other = self.input_W - new_w - self.x_shift

            new_h = int(self.img_h * self.y_scale)
            self.y_shift = (self.input_H - new_h) // 2
            y_other = self.input_H - new_h - self.y_shift

            input_tensor = cv2.resize(img, (new_w, new_h))
            input_tensor = cv2.copyMakeBorder(
                input_tensor,
                self.y_shift, y_other, self.x_shift, x_other,
                cv2.BORDER_CONSTANT,
                value=[127, 127, 127]
            )
            input_tensor = self.bgr2nv12(input_tensor)
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        return input_tensor

    def bgr2nv12(self, bgr_img):
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor):
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs):
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

    def postProcess(self, outputs):
        begin_time = time()
        clses = [
            outputs[0].reshape(-1, self.CLASSES_NUM),
            outputs[2].reshape(-1, self.CLASSES_NUM),
            outputs[4].reshape(-1, self.CLASSES_NUM)
        ]
        bboxes = [
            outputs[1].reshape(-1, self.REG * 4),
            outputs[3].reshape(-1, self.REG * 4),
            outputs[5].reshape(-1, self.REG * 4)
        ]

        dbboxes, ids, scores = [], [], []
        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):
            max_scores = np.max(cls, axis=1)
            bbox_selected = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            ids.append(np.argmax(cls[bbox_selected, :], axis=1))
            scores.append(1 / (1 + np.exp(-max_scores[bbox_selected])))
            ltrb_selected = np.sum(
                softmax(bbox[bbox_selected, :].reshape(-1, 4, self.REG), axis=2) * self.weights_static,
                axis=2
            )
            grid_selected = grid[bbox_selected, :]
            x1y1 = grid_selected - ltrb_selected[:, 0:2]
            x2y2 = grid_selected + ltrb_selected[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)

        dbboxes = np.concatenate((dbboxes), axis=0)
        scores = np.concatenate((scores), axis=0)
        ids = np.concatenate((ids), axis=0)
        hw = (dbboxes[:, 2:4] - dbboxes[:, 0:2])
        xyhw2 = np.hstack([dbboxes[:, 0:2], hw])

        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids == i
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices, :], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                x1, y1, x2, y2 = dbboxes[id_indices, :][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)

                x1 = x1 if x1 > 0 else 0
                x2 = x2 if x2 > 0 else 0
                y1 = y1 if y1 > 0 else 0
                y2 = y2 if y2 > 0 else 0
                x1 = x1 if x1 < self.img_w else self.img_w
                x2 = x2 if x2 < self.img_w else self.img_w
                y1 = y1 if y1 < self.img_h else self.img_h
                y2 = y2 if y2 < self.img_h else self.img_h

                score_val = float(scores[id_indices][indic])
                results.append((i, score_val, x1, y1, x2, y2))

        logger.debug("\033[1;31m" + f"Post Process time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        return results


coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
]


def draw_detection(img, bbox, score, class_id) -> None:
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
