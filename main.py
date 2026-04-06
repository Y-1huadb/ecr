#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import time

from vision.runtime.service import AstraVisionService
from vision.web.server import VisionWebServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astra + YOLO(.bin) + Web service")
    parser.add_argument("--host", default="0.0.0.0", help="Web host")
    parser.add_argument("--port", type=int, default=8080, help="Web port")
    parser.add_argument(
        "--model",
        default="models/yolov12n_detect_bayese_640x640_nv12_modified.bin",
        help="Path to Horizon YOLO .bin model",
    )
    parser.add_argument("--topic", default="/astra/color/image/compressed", help="ROS2 compressed image topic")
    parser.add_argument("--fps", type=int, default=30, help="Target processing FPS")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    service = AstraVisionService(
        model_path=args.model,
        topic_name=args.topic,
        fps=args.fps,
        min_confidence=args.confidence,
    )
    web = VisionWebServer(state=service.state, host=args.host, port=args.port)

    try:
        service.start()
        web.start()
        logging.info("Open http://%s:%s", args.host, args.port)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping")
    finally:
        web.close()
        service.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
