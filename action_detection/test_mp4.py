#!/usr/bin/env python

# Copyright (c) 2024, WuChao D-Robotics.
# Modified to support both image and MP4 video input/output.

import os
import cv2
import numpy as np

# scipy
try:
    from scipy.special import softmax
except:
    print("scipy is not installed, installing.")
    os.system("pip install scipy")
    from scipy.special import softmax

# hobot_dnn
try:
    try:
        from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
    except:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn  # BSP Python API from PyPI
except:
    print("pip install hobot-dnn-rdkx5")
    from hobot_dnn_rdkx5 import pyeasy_dnn as dnn

from time import time
import argparse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_POSE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,
                        default='/home/sunrise/Desktop/ECR/models/yolo11n_pose_bayese_640x640_nv12.bin',
                        help='Path to BPU Quantized *.bin Model.')
    parser.add_argument('--input', type=str,
                        default='/home/sunrise/Desktop/ECR/test.jpg',
                        help='Path to input image or video.')
    parser.add_argument('--output', type=str,
                        default='py_result.jpg',
                        help='Path to output image or video.')
    parser.add_argument('--classes-num', type=int, default=1, help='Classes Num to Detect.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--nkpt', type=int, default=17, help='num of keypoints.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='keypoint confidence threshold.')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))),
                        default=[8, 16, 32],
                        help='--strides 8,16,32')
    parser.add_argument('--show-fps', action='store_true', help='Overlay frame/inference/fps text on video.')
    opt = parser.parse_args()
    logger.info(opt)

    if not os.path.exists(opt.model_path):
        print(f"file {opt.model_path} does not exist. downloading ...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_pose_bayese_640x640_nv12.bin")
        opt.model_path = 'yolo11n_pose_bayese_640x640_nv12.bin'

    model = Ultralytics_YOLO_Pose_Bayese_YUV420SP(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        nms_thres=opt.nms_thres,
        score_thres=opt.score_thres,
        reg=opt.reg,
        strides=opt.strides,
        nkpt=opt.nkpt
    )

    suffix = os.path.splitext(opt.input)[1].lower()
    if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        infer_image(model, opt.input, opt.output, opt.kpt_conf_thres)
    elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.m4v']:
        infer_video(model, opt.input, opt.output, opt.kpt_conf_thres, opt.show_fps)
    else:
        raise ValueError(f"Unsupported input type: {opt.input}")


class Ultralytics_YOLO_Pose_Bayese_YUV420SP():
    def __init__(self, model_path, classes_num, nms_thres, score_thres, reg, strides, nkpt):
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_path)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms" % (1000 * (time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("❌ Failed to load model file: %s" % model_path)
            logger.error("You can download the model file from the following docs: ./models/download.md")
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_output in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_output.name}, type={quantize_output.properties.dtype}, shape={quantize_output.properties.shape}")

        self.REG = reg
        self.CLASSES_NUM = classes_num
        self.SCORE_THRESHOLD = score_thres
        self.NMS_THRESHOLD = nms_thres
        self.CONF_THRES_RAW = -np.log(1 / self.SCORE_THRESHOLD - 1)
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        self.strides = strides
        self.nkpt = nkpt
        logger.info(f"{self.REG = }, {self.CLASSES_NUM = }")
        logger.info("SCORE_THRESHOLD  = %.2f, NMS_THRESHOLD = %.2f" % (self.SCORE_THRESHOLD, self.NMS_THRESHOLD))
        logger.info("CONF_THRES_RAW = %.2f" % self.CONF_THRES_RAW)
        logger.info(f"{self.input_H = }, {self.input_W = }")
        logger.info(f"{self.strides = }")
        logger.info(f"{self.nkpt = }")

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
        logger.info(f"PREPROCESS_TYPE = {PREPROCESS_TYPE}")

        begin_time = time()
        self.img_h, self.img_w = img.shape[0:2]
        if PREPROCESS_TYPE == RESIZE_TYPE:
            begin_time = time()
            input_tensor = cv2.resize(img, (self.input_W, self.input_H), interpolation=cv2.INTER_NEAREST)
            input_tensor = self.bgr2nv12(input_tensor)
            self.y_scale = 1.0 * self.input_H / self.img_h
            self.x_scale = 1.0 * self.input_W / self.img_w
            self.y_shift = 0
            self.x_shift = 0
            logger.info("\033[1;31m" + f"pre process(resize) time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        elif PREPROCESS_TYPE == LETTERBOX_TYPE:
            begin_time = time()
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
            logger.info("\033[1;31m" + f"pre process(letter box) time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        else:
            logger.error(f"illegal PREPROCESS_TYPE = {PREPROCESS_TYPE}")
            exit(-1)

        logger.debug("\033[1;31m" + f"pre process time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        logger.info(f"y_scale = {self.y_scale:.2f}, x_scale = {self.x_scale:.2f}")
        logger.info(f"y_shift = {self.y_shift:.2f}, x_shift = {self.x_shift:.2f}")
        return input_tensor

    def bgr2nv12(self, bgr_img):
        begin_time = time()
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
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
        clses = [outputs[0].reshape(-1, self.CLASSES_NUM), outputs[3].reshape(-1, self.CLASSES_NUM), outputs[6].reshape(-1, self.CLASSES_NUM)]
        bboxes = [outputs[1].reshape(-1, self.REG * 4), outputs[4].reshape(-1, self.REG * 4), outputs[7].reshape(-1, self.REG * 4)]
        kpts = [outputs[2].reshape(-1, self.nkpt * 3), outputs[5].reshape(-1, self.nkpt * 3), outputs[8].reshape(-1, self.nkpt * 3)]

        dbboxes, ids, scores, kpts_xy, kpts_score = [], [], [], [], []
        for cls, bbox, stride, grid, kpt in zip(clses, bboxes, self.strides, self.grids, kpts):
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

            kpt = kpt[bbox_selected, :].reshape(-1, self.nkpt, 3)
            kpts_xy.append((kpt[:, :, :2] * 2.0 + (grid[bbox_selected, :][:, np.newaxis, :] - 0.5)) * stride)
            kpts_score.append(kpt[:, :, 2:3])

        dbboxes = np.concatenate(dbboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)
        hw = (dbboxes[:, 2:4] - dbboxes[:, 0:2])
        xyhw2 = np.hstack([dbboxes[:, 0:2], hw])

        kpts_xy = np.concatenate(kpts_xy, axis=0)
        kpts_score = np.concatenate(kpts_score, axis=0)

        results = []
        for i in range(self.CLASSES_NUM):
            id_indices = ids == i
            indices = cv2.dnn.NMSBoxes(xyhw2[id_indices, :], scores[id_indices], self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            if len(indices) == 0:
                continue
            for indic in indices:
                if isinstance(indic, (list, tuple, np.ndarray)):
                    indic = int(indic[0])
                x1, y1, x2, y2 = dbboxes[id_indices, :][indic]
                x1 = int((x1 - self.x_shift) / self.x_scale)
                y1 = int((y1 - self.y_shift) / self.y_scale)
                x2 = int((x2 - self.x_shift) / self.x_scale)
                y2 = int((y2 - self.y_shift) / self.y_scale)

                x1 = np.clip(x1, 0, self.img_w)
                x2 = np.clip(x2, 0, self.img_w)
                y1 = np.clip(y1, 0, self.img_h)
                y2 = np.clip(y2, 0, self.img_h)

                kpts_ = []
                for j in range(self.nkpt):
                    kpt_x = kpts_xy[id_indices, :][indic][j, 0]
                    kpt_y = kpts_xy[id_indices, :][indic][j, 1]
                    kpt_score = kpts_score[id_indices, :][indic][j, 0]

                    kpt_x = int((kpt_x - self.x_shift) / self.x_scale)
                    kpt_y = int((kpt_y - self.y_shift) / self.y_scale)
                    kpt_x = int(np.clip(kpt_x, 0, self.img_w))
                    kpt_y = int(np.clip(kpt_y, 0, self.img_h))
                    kpts_.append((kpt_x, kpt_y, kpt_score))

                results.append((i, scores[id_indices][indic], int(x1), int(y1), int(x2), int(y2), kpts_))
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

POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def draw_detection(img, bbox, score, class_id):
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_pose(img, kpts, kpt_conf_thres=0.5):
    kpt_conf_inverse = -np.log(1 / kpt_conf_thres - 1)

    for a, b in POSE_SKELETON:
        if a >= len(kpts) or b >= len(kpts):
            continue
        xa, ya, score_a = int(kpts[a][0]), int(kpts[a][1]), kpts[a][2]
        xb, yb, score_b = int(kpts[b][0]), int(kpts[b][1]), kpts[b][2]
        if score_a >= kpt_conf_inverse and score_b >= kpt_conf_inverse:
            cv2.line(img, (xa, ya), (xb, yb), (0, 255, 0), 2)

    for j in range(len(kpts)):
        x, y = int(kpts[j][0]), int(kpts[j][1])
        if kpts[j][2] < kpt_conf_inverse:
            continue
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def infer_single_frame(model, frame):
    input_tensor = model.preprocess_yuv420sp(frame)
    outputs = model.c2numpy(model.forward(input_tensor))
    return model.postProcess(outputs)


def render_results(frame, results, kpt_conf_thres):
    for class_id, score, x1, y1, x2, y2, kpts in results:
        draw_detection(frame, (x1, y1, x2, y2), score, class_id)
        draw_pose(frame, kpts, kpt_conf_thres)
    return frame


def infer_image(model, input_path, output_path, kpt_conf_thres):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Load image failed: {input_path}")

    results = infer_single_frame(model, img)
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    render_results(img, results, kpt_conf_thres)
    cv2.imwrite(output_path, img)
    logger.info("\033[1;32m" + f"saved in path: {output_path}" + "\033[0m")


def infer_video(model, input_path, output_path, kpt_conf_thres, show_fps=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Open video failed: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Open VideoWriter failed: {output_path}")

    frame_idx = 0
    total_infer = 0.0

    logger.info("\033[1;32m" + f"Processing video: {input_path}" + "\033[0m")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time()
        results = infer_single_frame(model, frame)
        infer_ms = (time() - t0) * 1000.0
        total_infer += infer_ms

        render_results(frame, results, kpt_conf_thres)

        if show_fps:
            avg_fps = 1000.0 / (total_infer / max(frame_idx + 1, 1)) if total_infer > 0 else 0.0
            text = f"frame={frame_idx} infer={infer_ms:.1f}ms avg_fps={avg_fps:.2f}"
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        if frame_idx % 10 == 0:
            logger.info(f"processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("\033[1;32m" + f"saved video in path: {output_path}" + "\033[0m")


if __name__ == "__main__":
    main()
