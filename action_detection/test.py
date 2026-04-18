#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch



def load_model(weights_path: str, device: str = 'cpu', num_class: int = 400):
    from stgcn.st_gcn import Model

    model = Model(
        in_channels=3,
        num_class=num_class,
        edge_importance_weighting=True,
        graph_args={'layout': 'openpose', 'strategy': 'spatial'}
    )

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith('module.') else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    model.eval().to(device)
    return model, missing, unexpected


def load_labels(label_path: str):
    text = Path(label_path).read_text(encoding='utf-8').strip()
    labels = text.splitlines()
    if len(labels) == 1 and '\t' in labels[0]:
        labels = [x.strip() for x in labels[0].split('\t') if x.strip()]
    return labels


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
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


def normalize_like_demo(kpts18: np.ndarray, width: int, height: int) -> np.ndarray:
    out = kpts18.copy().astype(np.float32)
    out[:, :, 0] = out[:, :, 0] / float(width)
    out[:, :, 1] = out[:, :, 1] / float(height)
    out[:, :, 0:2] = out[:, :, 0:2] - 0.5
    zero_mask = out[:, :, 2] <= 0
    out[:, :, 0][zero_mask] = 0
    out[:, :, 1][zero_mask] = 0
    return out


def temporal_sample(seq: np.ndarray, out_len: int) -> np.ndarray:
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


def build_sample_from_track(track_kpts17: np.ndarray, width: int, height: int, window_size: int = 128):
    op18 = coco17_to_openpose18(track_kpts17)
    norm = normalize_like_demo(op18, width, height)
    seq = temporal_sample(norm, window_size)
    return seq.transpose(2, 0, 1)[..., None].astype(np.float32)


def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def predict_topk(model, data: np.ndarray, device: str, labels, topk: int = 5):
    tensor = torch.from_numpy(data).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(tensor)[0].detach().cpu().numpy()
    prob = softmax_np(logits)
    order = np.argsort(prob)[::-1][:topk]
    results = []
    for idx in order:
        label = labels[idx] if idx < len(labels) else str(idx)
        results.append((int(idx), label, float(prob[idx])))
    return logits, results


def sliding_windows(frames: np.ndarray, kpts17: np.ndarray, width: int, height: int,
                    window_size: int, stride: int):
    n = len(frames)
    if n == 0:
        return []
    if n <= window_size:
        return [(0, n, int(frames[0]), int(frames[-1]), build_sample_from_track(kpts17, width, height, window_size))]
    windows = []
    starts = list(range(0, n - window_size + 1, stride))
    if starts[-1] != n - window_size:
        starts.append(n - window_size)
    for s in starts:
        e = s + window_size
        sample = build_sample_from_track(kpts17[s:e], width, height, window_size)
        windows.append((s, e, int(frames[s]), int(frames[e - 1]), sample))
    return windows


def pick_track_label(model, labels, frames, kpts17, width, height, device, window_size, stride, topk, mode='window_avg'):
    whole = build_sample_from_track(kpts17, width, height, window_size)
    _, whole_topk = predict_topk(model, whole, device, labels, topk)
    if mode == 'whole_track':
        return whole_topk[0], {'whole_track_topk': whole_topk, 'window_avg_topk': whole_topk, 'windows': []}

    windows = sliding_windows(frames, kpts17, width, height, window_size, stride)
    agg_logits = []
    win_info = []
    for _, _, start_f, end_f, sample in windows:
        logits, topk_list = predict_topk(model, sample, device, labels, topk)
        agg_logits.append(logits)
        win_info.append({'start_frame': start_f, 'end_frame': end_f, 'topk': topk_list})

    if agg_logits:
        avg_prob = softmax_np(np.mean(np.stack(agg_logits, axis=0), axis=0))
        order = np.argsort(avg_prob)[::-1][:topk]
        avg_topk = [(int(i), labels[i] if i < len(labels) else str(i), float(avg_prob[i])) for i in order]
        return avg_topk[0], {'whole_track_topk': whole_topk, 'window_avg_topk': avg_topk, 'windows': win_info}
    return whole_topk[0], {'whole_track_topk': whole_topk, 'window_avg_topk': whole_topk, 'windows': []}


def make_frame_track_lookup(data):
    lookup = {}
    track_ids = [int(x) for x in data['track_ids']]
    for tid in track_ids:
        prefix = f'track_{tid:04d}'
        frames = data[f'{prefix}_frames']
        bboxes = data[f'{prefix}_bboxes']
        for f, box in zip(frames, bboxes):
            fi = int(f)
            lookup.setdefault(fi, []).append((tid, box.astype(np.int32)))
    return lookup


def draw_text_box(img, text, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, int(x))
    y = max(th + 4, int(y))
    x2 = min(img.shape[1] - 1, x + tw + 6)
    y2 = min(img.shape[0] - 1, y + 4)
    cv2.rectangle(img, (x, y - th - 6), (x2, y2), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def overlay_video(input_video, output_video, frame_lookup, track_best, show_top1_only=True):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {input_video}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to open VideoWriter: {output_video}')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, box in frame_lookup.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in box]
            best = track_best.get(tid)
            if best is None:
                continue
            _, label, score = best
            text = f'ID {tid} | {label} {score:.2f}' if show_top1_only else f'ID {tid}'
            draw_text_box(frame, text, x1, max(18, y1 - 8))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(description='Overlay ST-GCN action labels onto tracked pose video')
    ap.add_argument('--track-npz', default='/home/sunrise/Desktop/data/pose.npz', help='Path to per-track npz from yolo26_pose_track_npz.py')
    ap.add_argument('--weights', default='/home/sunrise/Desktop/ECR/models/st_gcn.kinetics.pt', help='Path to st_gcn.kinetics.pt or your own trained weights')
    ap.add_argument('--input-video', default='/home/sunrise/Desktop/data/-_pn5NxJmok_000004_000014.mp4', help='Source mp4 to draw action labels on')
    ap.add_argument('--output-video', default='/home/sunrise/Desktop/data/stgcn-result.mp4', help='Output mp4 with overlaid action labels')
    ap.add_argument('--label-path', default='/home/sunrise/Desktop/ECR/action_detection/stgcn/label_name.txt', help='Optional label txt. Default: ./stgcn/label_name.txt')
    ap.add_argument('--device', default='cpu', help='cpu or cuda:0')
    ap.add_argument('--window-size', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--prediction-mode', choices=['whole_track', 'window_avg'], default='window_avg')
    args = ap.parse_args()

    label_path = args.label_path or os.path.join('.', 'stgcn', 'label_name.txt')
    labels = load_labels(label_path)

    data = np.load(args.track_npz, allow_pickle=True)
    width = int(data['width'])
    height = int(data['height'])
    track_ids = [int(x) for x in data['track_ids']]

    model, missing, unexpected = load_model(args.weights, args.device, num_class=len(labels))
    if missing:
        print(f'[warn] missing keys: {len(missing)}')
    if unexpected:
        print(f'[warn] unexpected keys: {len(unexpected)}')

    track_best = {}
    for tid in track_ids:
        prefix = f'track_{tid:04d}'
        frames = data[f'{prefix}_frames']
        kpts17 = data[f'{prefix}_keypoints']
        if kpts17.ndim != 3 or kpts17.shape[1:] != (17, 3):
            continue
        best, extra = pick_track_label(
            model=model,
            labels=labels,
            frames=frames,
            kpts17=kpts17,
            width=width,
            height=height,
            device=args.device,
            window_size=args.window_size,
            stride=args.stride,
            topk=args.topk,
            mode=args.prediction_mode,
        )
        track_best[tid] = best
        print(f'track {tid:04d}: {best[1]} ({best[2]:.4f})')

    frame_lookup = make_frame_track_lookup(data)
    overlay_video(args.input_video, args.output_video, frame_lookup, track_best, show_top1_only=True)
    print(f'saved: {args.output_video}')


if __name__ == '__main__':
    main()
