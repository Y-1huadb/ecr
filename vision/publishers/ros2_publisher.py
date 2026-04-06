from __future__ import annotations

import logging
import threading

import cv2

LOGGER = logging.getLogger("astra_vision")

ROS2_AVAILABLE = False
CompressedImage = None
rclpy = None
Node = None
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from sensor_msgs.msg import CompressedImage  # type: ignore

    ROS2_AVAILABLE = True
except Exception:
    ROS2_AVAILABLE = False


class RosCompressedImagePublisher:
    def __init__(self, topic_name: str) -> None:
        self.enabled = False
        self._node = None
        self._publisher = None
        self._lock = threading.Lock()
        self._published_count = 0

        if not ROS2_AVAILABLE:
            LOGGER.warning("ROS2 unavailable, RGB topic publish disabled")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node("astra_vision_publisher")
        self._publisher = self._node.create_publisher(CompressedImage, topic_name, 10)
        self.enabled = True
        LOGGER.info("Publishing RGB stream to %s", topic_name)

    def publish(self, frame_bgr):
        if not self.enabled or self._publisher is None:
            return

        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return

        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self._publisher.publish(msg)

        with self._lock:
            self._published_count += 1

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if ROS2_AVAILABLE and rclpy is not None and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

    @property
    def published_count(self) -> int:
        with self._lock:
            return self._published_count
