#!/usr/bin/env python3
"""Evaluate an RDK X5 YOLO detection bin model on a subset of COCO val2017.

This script reuses the detector class from an existing yolo26_det.py-style file and
computes COCO metrics with pycocotools on only a subset of images (default: 100).

Example:
python3 eval_yolo26_coco100.py \
  --det-script /home/sunrise/Desktop/ECR/yolo26_det.py \
  --model-path /home/sunrise/Desktop/ECR/models/yolov12n_detect_bayese_640x640_nv12_modified.bin \
  --images-dir /data/coco/val2017 \
  --ann-file /data/coco/annotations/instances_val2017.json \
  --max-images 100 \
  --sample-mode first
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:
    raise SystemExit(
        "pycocotools is required. Install it with: pip install pycocotools"
    ) from e

# COCO 80-class index (model output) -> official COCO category id
COCO80_TO_CAT_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def load_detector_module(det_script: str):
    det_script = os.path.abspath(det_script)
    if not os.path.exists(det_script):
        raise FileNotFoundError(f"Detector script not found: {det_script}")

    spec = importlib.util.spec_from_file_location("user_yolo_det", det_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load detector script: {det_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_yolo_det"] = module
    spec.loader.exec_module(module)
    return module


def build_detector(module, args):
    cls_name = "Ultralytics_YOLO_Detect_Bayese_YUV420SP"
    if not hasattr(module, cls_name):
        raise AttributeError(
            f"{args.det_script} does not define {cls_name}."
        )
    detector_cls = getattr(module, cls_name)
    return detector_cls(
        model_path=args.model_path,
        classes_num=args.classes_num,
        nms_thres=args.nms_thres,
        score_thres=args.score_thres,
        reg=args.reg,
        strides=args.strides,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO bin model on 100 COCO val images"
    )
    parser.add_argument(
        "--det-script",
        type=str,
        default="object_detection/yolo26_det.py",
        help="Path to your existing detection script.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/sunrise/Desktop/ECR/models/yolov12n_detect_bayese_640x640_nv12_modified.bin",
        help="Path to BPU quantized .bin model.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/home/sunrise/Desktop/ECR/data/images",
        help="COCO val2017 image directory.",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="/home/sunrise/Desktop/ECR/data/annotations/instances_val2017_100.json",
        help="COCO instances_val2017.json path.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Number of images to evaluate. Default: 100",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="Use first N images or random N images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when sample-mode=random.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="/home/sunrise/Desktop/ECR/eval/det/eval_det_results.json",
        help="Optional path to save detection results in COCO json format.",
    )
    parser.add_argument('--classes-num', type=int, default=80)
    parser.add_argument('--nms-thres', type=float, default=0.7)
    parser.add_argument('--score-thres', type=float, default=0.25)
    parser.add_argument('--reg', type=int, default=16)
    parser.add_argument(
        '--strides',
        type=lambda s: list(map(int, s.split(','))),
        default=[8, 16, 32],
        help='--strides 8,16,32'
    )
    return parser.parse_args()


def select_image_ids(coco_gt: COCO, max_images: int, sample_mode: str, seed: int) -> List[int]:
    image_ids = sorted(coco_gt.getImgIds())
    if max_images <= 0 or max_images >= len(image_ids):
        return image_ids
    if sample_mode == "first":
        return image_ids[:max_images]
    rng = random.Random(seed)
    return sorted(rng.sample(image_ids, max_images))


def infer_one_image(detector, img_bgr: np.ndarray):
    input_tensor = detector.preprocess_yuv420sp(img_bgr)
    outputs = detector.c2numpy(detector.forward(input_tensor))
    return detector.postProcess(outputs)


def results_to_coco(image_id: int, results: Sequence[Tuple[int, float, int, int, int, int]]):
    coco_results = []
    for class_id, score, x1, y1, x2, y2 in results:
        if class_id < 0 or class_id >= len(COCO80_TO_CAT_ID):
            continue
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        if w <= 0 or h <= 0:
            continue
        coco_results.append({
            "image_id": int(image_id),
            "category_id": int(COCO80_TO_CAT_ID[int(class_id)]),
            "bbox": [float(x1), float(y1), w, h],
            "score": float(score),
        })
    return coco_results


def main():
    args = parse_args()

    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"annotation file not found: {args.ann_file}")

    print("[INFO] Loading detector module...")
    module = load_detector_module(args.det_script)
    print("[INFO] Building detector...")
    detector = build_detector(module, args)

    print("[INFO] Loading COCO annotations...")
    coco_gt = COCO(args.ann_file)
    image_ids = select_image_ids(coco_gt, args.max_images, args.sample_mode, args.seed)
    print(f"[INFO] Evaluating {len(image_ids)} images (sample_mode={args.sample_mode})")

    all_results: List[Dict] = []
    total_infer = 0.0
    total_images = 0

    for idx, image_id in enumerate(image_ids, start=1):
        img_info = coco_gt.loadImgs([image_id])[0]
        img_path = os.path.join(args.images_dir, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        t0 = time.time()
        det_results = infer_one_image(detector, img)
        infer_ms = (time.time() - t0) * 1000.0

        all_results.extend(results_to_coco(image_id, det_results))
        total_infer += infer_ms
        total_images += 1

        if idx % 10 == 0 or idx == len(image_ids):
            avg_ms = total_infer / max(total_images, 1)
            print(
                f"[INFO] {idx}/{len(image_ids)} | avg infer {avg_ms:.2f} ms/img | "
                f"detections so far: {len(all_results)}"
            )

    if total_images == 0:
        raise RuntimeError("No images were evaluated.")

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f)
        print(f"[INFO] Saved COCO detections to: {save_path}")

    if len(all_results) == 0:
        print("[WARN] Model produced no detections. COCOeval cannot run meaningfully.")
        return

    coco_dt = coco_gt.loadRes(all_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    print("\n[RESULT] Key metrics on selected subset:")
    print(f"AP@[0.50:0.95] = {stats[0]:.4f}")
    print(f"AP@0.50        = {stats[1]:.4f}")
    print(f"AP@0.75        = {stats[2]:.4f}")
    print(f"AR@[1]         = {stats[6]:.4f}")
    print(f"AR@[10]        = {stats[7]:.4f}")
    print(f"AR@[100]       = {stats[8]:.4f}")
    print(f"Avg infer time = {total_infer / total_images:.2f} ms/img over {total_images} images")
    print("\n[NOTE] 100-image COCO results are only a quick estimate, not the official val2017 metric.")


if __name__ == "__main__":
    main()
