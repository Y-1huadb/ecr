#!/usr/bin/env python3
"""
ST-GCN BPU 推理：支持 Kinetics skeleton JSON，或 YOLO 轨迹 npz（COCO-17，在脚本内转为 OpenPose-18）。
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class STGCNRunner:
    def __init__(self, so_path: str = "./libstgcn_hbdnn.so", model_path: str = "./stgcn_x5.bin"):
        self.lib = ctypes.CDLL(so_path)

        self.lib.stgcn_create.argtypes = [ctypes.c_char_p]
        self.lib.stgcn_create.restype = ctypes.c_void_p

        self.lib.stgcn_get_input_numel.argtypes = [ctypes.c_void_p]
        self.lib.stgcn_get_input_numel.restype = ctypes.c_int

        self.lib.stgcn_get_output_numel.argtypes = [ctypes.c_void_p]
        self.lib.stgcn_get_output_numel.restype = ctypes.c_int

        self.lib.stgcn_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.stgcn_run.restype = ctypes.c_int

        self.lib.stgcn_destroy.argtypes = [ctypes.c_void_p]
        self.lib.stgcn_destroy.restype = None

        self.handle = self.lib.stgcn_create(model_path.encode("utf-8"))
        if not self.handle:
            raise RuntimeError("stgcn_create failed")

        self.input_numel = self.lib.stgcn_get_input_numel(self.handle)
        self.output_numel = self.lib.stgcn_get_output_numel(self.handle)

    def run(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x)

        if x.size != self.input_numel:
            raise ValueError(f"input numel mismatch: got {x.size}, expect {self.input_numel}")

        y = np.empty((self.output_numel,), dtype=np.float32)

        ret = self.lib.stgcn_run(
            self.handle,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.size,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.size,
        )
        if ret != 0:
            raise RuntimeError(f"stgcn_run failed, ret={ret}")
        return y

    def close(self) -> None:
        if self.handle:
            self.lib.stgcn_destroy(self.handle)
            self.handle = None


def temporal_sample_or_pad_ctv(x_ctv: np.ndarray, window_size: int) -> np.ndarray:
    """x_ctv: [C, T, V]"""
    _, t, _ = x_ctv.shape
    if t == window_size:
        return x_ctv
    if t > window_size:
        start = (t - window_size) // 2
        return x_ctv[:, start : start + window_size, :]
    pad_num = window_size - t
    pad = np.repeat(x_ctv[:, -1:, :], pad_num, axis=1)
    return np.concatenate([x_ctv, pad], axis=1)


def temporal_interpolate_tvj(seq: np.ndarray, out_len: int) -> np.ndarray:
    """(T, V, C) -> (out_len, V, C)，与 test.py 中 temporal_sample 行为一致。"""
    t = seq.shape[0]
    if t == out_len:
        return seq.astype(np.float32)
    if t == 0:
        return np.zeros((out_len,) + seq.shape[1:], dtype=np.float32)
    if t == 1:
        return np.repeat(seq, out_len, axis=0).astype(np.float32)
    src = np.linspace(0, t - 1, t)
    dst = np.linspace(0, t - 1, out_len)
    out = np.zeros((out_len,) + seq.shape[1:], dtype=np.float32)
    flat_dim = int(np.prod(seq.shape[1:]))
    seq2 = seq.reshape(t, flat_dim)
    out2 = out.reshape(out_len, flat_dim)
    for i in range(flat_dim):
        out2[:, i] = np.interp(dst, src, seq2[:, i])
    return out


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """(T, 17, 3) -> (T, 18, 3)，与 ST-GCN openpose 图结构一致。"""
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.ndim != 3 or kpts17.shape[1:] != (17, 3):
        raise ValueError(f"expected (T, 17, 3), got {kpts17.shape}")

    out = np.zeros((kpts17.shape[0], 18, 3), dtype=np.float32)
    out[:, 0] = kpts17[:, 0]
    out[:, 2] = kpts17[:, 6]
    out[:, 3] = kpts17[:, 8]
    out[:, 4] = kpts17[:, 10]
    out[:, 5] = kpts17[:, 5]
    out[:, 6] = kpts17[:, 7]
    out[:, 7] = kpts17[:, 9]
    out[:, 8] = kpts17[:, 12]
    out[:, 9] = kpts17[:, 14]
    out[:, 10] = kpts17[:, 16]
    out[:, 11] = kpts17[:, 11]
    out[:, 12] = kpts17[:, 13]
    out[:, 13] = kpts17[:, 15]
    out[:, 14] = kpts17[:, 2]
    out[:, 15] = kpts17[:, 1]
    out[:, 16] = kpts17[:, 4]
    out[:, 17] = kpts17[:, 3]

    ls = kpts17[:, 5]
    rs = kpts17[:, 6]
    neck_xy = (ls[:, :2] + rs[:, :2]) / 2.0
    neck_score = np.minimum(ls[:, 2], rs[:, 2])
    valid = (ls[:, 2] > 0) & (rs[:, 2] > 0)
    out[:, 1, :2] = neck_xy
    out[:, 1, 2] = neck_score
    out[:, 1, :2] *= valid[:, None].astype(np.float32)
    out[:, 1, 2] *= valid.astype(np.float32)
    return out


def normalize_pixel_openpose18(kpts18: np.ndarray, width: int, height: int) -> np.ndarray:
    """像素坐标 (T,18,3) -> 与 test.py normalize_like_demo 一致。"""
    out = kpts18.copy().astype(np.float32)
    out[:, :, 0] = out[:, :, 0] / float(width)
    out[:, :, 1] = out[:, :, 1] / float(height)
    out[:, :, 0:2] = out[:, :, 0:2] - 0.5
    zero_mask = out[:, :, 2] <= 0
    out[:, :, 0][zero_mask] = 0
    out[:, :, 1][zero_mask] = 0
    return out


def track_keypoints_to_stgcn_input(
    keypoints_tvj: np.ndarray,
    width: int,
    height: int,
    window_size: int,
    pad_or_interpolate: str = "interpolate",
) -> np.ndarray:
    """
    npz 中每帧关键点 -> BPU ST-GCN 输入 [1, 3, window_size, 18]。

    keypoints_tvj: (T, J, 3)，J 为 17（COCO）或 18（已是 OpenPose）。
    pad_or_interpolate:
      - "interpolate": 时间维插值到 window_size（与 test.py 一致）
      - "pad": 与 JSON 分支相同，用 temporal_sample_or_pad_ctv（中心裁切或末尾填充）
    """
    keypoints_tvj = np.asarray(keypoints_tvj, dtype=np.float32)
    if keypoints_tvj.ndim != 3 or keypoints_tvj.shape[2] != 3:
        raise ValueError(f"expected (T, J, 3), got {keypoints_tvj.shape}")

    j = keypoints_tvj.shape[1]
    if j == 17:
        op18 = coco17_to_openpose18(keypoints_tvj)
    elif j == 18:
        op18 = keypoints_tvj
    else:
        raise ValueError(f"expected 17 (COCO) or 18 (OpenPose) joints, got J={j}")

    norm = normalize_pixel_openpose18(op18, width, height)
    # (T, 18, 3) -> (3, T, 18)
    ctv = np.transpose(norm, (2, 0, 1)).astype(np.float32)

    if pad_or_interpolate == "pad":
        ctv = temporal_sample_or_pad_ctv(ctv, window_size)
    elif pad_or_interpolate == "interpolate":
        # (3, T, 18) -> (T, 18, 3) 插值再转回
        tvj = np.transpose(ctv, (1, 2, 0))
        tvj = temporal_interpolate_tvj(tvj, window_size)
        ctv = np.transpose(tvj, (2, 0, 1)).astype(np.float32)
    else:
        raise ValueError("pad_or_interpolate must be 'pad' or 'interpolate'")

    inp = ctv[None, ...].astype(np.float32)
    return np.ascontiguousarray(inp)


def load_kinetics_skeleton_json(
    json_path: str, v: int = 18, frame_index_starts_from_one: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """读取单个 kinetics-skeleton json，返回 [C, T, V, M] 与 meta。"""
    with open(json_path, "r", encoding="utf-8") as f:
        video_info = json.load(f)

    frames = video_info.get("data", [])
    if not frames:
        raise ValueError(f"json 中 data 为空: {json_path}")

    max_frame_index = 0
    max_person = 1
    for frame_info in frames:
        frame_index = int(frame_info.get("frame_index", 0))
        max_frame_index = max(max_frame_index, frame_index)
        max_person = max(max_person, len(frame_info.get("skeleton", [])))

    if max_frame_index <= 0:
        raise ValueError(f"json 中没有有效 frame_index: {json_path}")

    c = 3
    t = max_frame_index
    m = max_person
    data_numpy = np.zeros((c, t, v, m), dtype=np.float32)

    for frame_info in frames:
        frame_index = int(frame_info.get("frame_index", 0))
        if frame_index_starts_from_one:
            frame_index -= 1

        if frame_index < 0 or frame_index >= t:
            continue

        for person_i, skeleton_info in enumerate(frame_info.get("skeleton", [])[:m]):
            pose = skeleton_info.get("pose", [])
            score = skeleton_info.get("score", [])

            if len(pose) != 2 * v or len(score) != v:
                continue

            data_numpy[0, frame_index, :, person_i] = np.asarray(pose[0::2], dtype=np.float32)
            data_numpy[1, frame_index, :, person_i] = np.asarray(pose[1::2], dtype=np.float32)
            data_numpy[2, frame_index, :, person_i] = np.asarray(score, dtype=np.float32)

    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    return data_numpy, video_info


def select_person(sample_ctvm: np.ndarray, mode: str = "best") -> Tuple[np.ndarray, int]:
    """sample_ctvm: [C, T, V, M] -> ([C, T, V], person_idx)"""
    _, _, _, m = sample_ctvm.shape
    if m == 1:
        return sample_ctvm[..., 0], 0
    if mode == "first":
        return sample_ctvm[..., 0], 0

    scores = sample_ctvm[2].sum(axis=(0, 1))
    idx = int(np.argmax(scores))
    return sample_ctvm[..., idx], idx


def build_stgcn_input_from_json(
    json_path: str,
    window_size: int = 32,
    person_mode: str = "best",
    v: int = 18,
) -> Tuple[np.ndarray, Dict[str, Any], int]:
    sample_ctvm, meta = load_kinetics_skeleton_json(json_path, v=v)
    ctv, chosen_person = select_person(sample_ctvm, mode=person_mode)
    ctv = temporal_sample_or_pad_ctv(ctv, window_size)
    inp = ctv[None, ...].astype(np.float32)
    inp = np.ascontiguousarray(inp)
    return inp, meta, chosen_person


def load_labels(label_path: Optional[str]) -> List[str]:
    if not label_path:
        return []
    p = Path(label_path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8").strip()
    labels = text.splitlines()
    if len(labels) == 1 and "\t" in labels[0]:
        labels = [x.strip() for x in labels[0].split("\t") if x.strip()]
    return labels


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def run_npz_tracks(
    npz_path: str,
    window_size: int,
    pad_or_interpolate: str,
) -> List[Dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    width = int(data["width"])
    height = int(data["height"])
    track_ids = [int(x) for x in data["track_ids"]]

    tracks_out: List[Dict[str, Any]] = []
    for tid in track_ids:
        prefix = f"track_{tid:04d}"
        coco_key = f"{prefix}_keypoints"
        op_key = f"{prefix}_keypoints_openpose18"
        if op_key in data.files:
            kpts = np.asarray(data[op_key], dtype=np.float32)
            layout = "openpose18_saved"
        else:
            kpts = np.asarray(data[coco_key], dtype=np.float32)
            layout = "coco17"

        if kpts.ndim != 3 or kpts.shape[2] != 3 or kpts.shape[1] not in (17, 18):
            continue

        frames = np.asarray(data[f"{prefix}_frames"], dtype=np.int32)
        inp = track_keypoints_to_stgcn_input(
            kpts, width, height, window_size, pad_or_interpolate=pad_or_interpolate
        )

        tracks_out.append(
            {
                "track_id": int(tid),
                "keypoint_layout_in_npz": layout,
                "converted_to_openpose18": layout == "coco17",
                "num_frames": int(kpts.shape[0]),
                "start_frame": int(frames[0]) if len(frames) else None,
                "end_frame": int(frames[-1]) if len(frames) else None,
                "stgcn_input_shape": list(inp.shape),
                "input": inp,
            }
        )
    return tracks_out


def build_result_dict(
    logits: np.ndarray,
    labels: List[str],
    topk: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    topk = min(topk, logits.shape[0])
    prob = softmax_np(logits)
    order = np.argsort(prob)[::-1][:topk]
    top_list = []
    for idx in order:
        lab = labels[int(idx)] if labels and int(idx) < len(labels) else str(int(idx))
        top_list.append({"class_id": int(idx), "label": lab, "prob": float(prob[int(idx)])})
    out = {
        "top1": top_list[0] if top_list else None,
        "topk": top_list,
        "logits": logits.astype(np.float32).tolist(),
    }
    out.update(extra)
    return out


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_labels = here / "stgcn" / "label_name.txt"

    p = argparse.ArgumentParser(description="ST-GCN BPU：Kinetics JSON 或 YOLO 轨迹 npz（COCO17→OpenPose18）")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=str, default=None, help="Kinetics skeleton JSON 路径")
    src.add_argument("--track-npz", type=str, default=None, help="test_pose_mp4_save_npz 等保存的轨迹 npz")

    p.add_argument("--so-path", type=str, default=str(here / "libstgcn_hbdnn.so"))
    p.add_argument("--model-path", type=str, default=str(here.parent / "models" / "stgcn_x5.bin"))
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--person-mode", choices=["best", "first"], default="best", help="JSON 多行人时选人策略")
    p.add_argument(
        "--time-align",
        choices=["interpolate", "pad"],
        default="interpolate",
        help="npz 时间维对齐到 window：与 test.py 一致用 interpolate；pad 与 JSON 分支一致",
    )
    p.add_argument("--label-path", type=str, default=str(default_labels), help="Kinetics 类别名 txt，可选")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="将推理结果写入该路径（默认不写完整 logits 以减小体积）；不传则只打印",
    )
    p.add_argument(
        "--include-logits",
        action="store_true",
        help="写入 --output-json 时保留每条结果的完整 logits 向量",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.label_path if args.label_path else None)

    runner = STGCNRunner(so_path=args.so_path, model_path=args.model_path)
    results_root: Dict[str, Any] = {
        "model_path": args.model_path,
        "so_path": args.so_path,
        "window_size": int(args.window_size),
    }

    try:
        if args.json:
            inp, meta, chosen = build_stgcn_input_from_json(
                json_path=args.json,
                window_size=args.window_size,
                person_mode=args.person_mode,
                v=18,
            )
            logits = runner.run(inp)
            print("input shape:", inp.shape)
            print("chosen_person:", chosen)

            res = build_result_dict(
                logits,
                labels,
                args.topk,
                {"source": "json", "json_path": args.json, "chosen_person": int(chosen)},
            )
            results_root["single"] = res

        else:
            tracks = run_npz_tracks(args.track_npz, args.window_size, args.time_align)
            results_root["source"] = "track_npz"
            results_root["track_npz"] = args.track_npz
            results_root["tracks"] = []

            for item in tracks:
                inp = item.pop("input")
                logits = runner.run(inp)
                print(f"track {item['track_id']:04d} input {tuple(inp.shape)} -> logits {logits.shape}")

                res = build_result_dict(
                    logits,
                    labels,
                    args.topk,
                    {k: v for k, v in item.items() if k != "input"},
                )
                results_root["tracks"].append(res)

    finally:
        runner.close()

    out_path = (args.output_json or "").strip()
    if out_path:
        to_save = dict(results_root)
        if not args.include_logits:

            def strip_logits(d: Dict[str, Any]) -> Dict[str, Any]:
                d = dict(d)
                d.pop("logits", None)
                return d

            if "single" in to_save:
                to_save["single"] = strip_logits(to_save["single"])
            if "tracks" in to_save:
                to_save["tracks"] = [strip_logits(t) for t in to_save["tracks"]]

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out_path}" + ("" if args.include_logits else " (without full logits; use --include-logits to embed them)"))

    # 控制台打印 top1
    if "single" in results_root:
        r = results_root["single"]
        if r.get("top1"):
            print("top1:", r["top1"])
    if "tracks" in results_root:
        for r in results_root["tracks"]:
            if r.get("top1"):
                tid = r.get("track_id", "?")
                print(f"track {tid} top1:", r["top1"])


if __name__ == "__main__":
    main()
