#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from test import load_labels, load_model, pick_track_label


def main():
    ap = argparse.ArgumentParser(
        description='Export top-k action predictions for each track from track npz'
    )
    ap.add_argument(
        '--track-npz',
        default='/home/sunrise/Desktop/data/pose.npz',
        help='Path to per-track npz from pose tracking script'
    )
    ap.add_argument(
        '--weights',
        default='/home/sunrise/Desktop/ECR/models/st_gcn.kinetics.pt',
        help='Path to st_gcn weights'
    )
    ap.add_argument(
        '--label-path',
        default='/home/sunrise/Desktop/ECR/action_detection/stgcn/label_name.txt',
        help='Path to label txt'
    )
    ap.add_argument('--device', default='c、pu', help='cpu or cuda:0')
    ap.add_argument('--window-size', type=int, default=32)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument(
        '--prediction-mode',
        choices=['whole_track', 'window_avg'],
        default='window_avg'
    )
    ap.add_argument(
        '--output-json',
        default='/home/sunrise/Desktop/data/track_top5.json',
        help='Output path for per-track top-k JSON'
    )
    args = ap.parse_args()

    labels = load_labels(args.label_path)
    model, missing, unexpected = load_model(
        args.weights, args.device, num_class=len(labels)
    )
    if missing:
        print(f'[warn] missing keys: {len(missing)}')
    if unexpected:
        print(f'[warn] unexpected keys: {len(unexpected)}')

    data = np.load(args.track_npz, allow_pickle=True)
    width = int(data['width'])
    height = int(data['height'])
    track_ids = [int(x) for x in data['track_ids']]

    results = {
        'track_npz': args.track_npz,
        'weights': args.weights,
        'label_path': args.label_path,
        'window_size': int(args.window_size),
        'stride': int(args.stride),
        'topk': int(args.topk),
        'prediction_mode': args.prediction_mode,
        'tracks': []
    }

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

        topk_list = extra['window_avg_topk']
        track_item = {
            'track_id': int(tid),
            'num_frames': int(len(frames)),
            'start_frame': int(frames[0]) if len(frames) > 0 else None,
            'end_frame': int(frames[-1]) if len(frames) > 0 else None,
            'topk': [
                {
                    'class_id': int(cls_id),
                    'label': label,
                    'prob': float(prob),
                }
                for cls_id, label, prob in topk_list
            ],
            'best': {
                'class_id': int(best[0]),
                'label': best[1],
                'prob': float(best[2]),
            },
        }
        results['tracks'].append(track_item)

        top5_text = ' | '.join(
            [f'{idx + 1}.{item["label"]}({item["prob"]:.4f})'
             for idx, item in enumerate(track_item['topk'])]
        )
        print(f'track {tid:04d}: {top5_text}')

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
