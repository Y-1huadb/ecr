#!/usr/bin/env python3
"""
Astra Pro + Web Server - RGB + Depth overlay display v2
Fixed alignment and transparency adjustment
"""

import ctypes
import os
import numpy as np
import cv2


MAX_DIST = 8000
INVALID_COLOR = [0, 0, 0]
print(f"[INFO] Parameters: max_dist=8000mm, invalid_color=Black")

# Astra SDK path setup
ASTRA_SDK_PATH = '/home/sunrise/Desktop/AstraSDK'
ASTRA_LIB_PATH = f'{ASTRA_SDK_PATH}/lib'

# Environment variables must be set before loading libraries
os.environ['DISPLAY'] = ':0'
os.environ['LD_LIBRARY_PATH'] = f"{ASTRA_LIB_PATH}:{ASTRA_LIB_PATH}/Plugins/openni2:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Load Astra SDK libraries
ctypes.CDLL(f'{ASTRA_LIB_PATH}/libastra_core.so', ctypes.RTLD_GLOBAL)
astra_core = ctypes.CDLL(f'{ASTRA_LIB_PATH}/libastra_core.so')
astra = ctypes.CDLL(f'{ASTRA_LIB_PATH}/libastra.so')

# Define types
astra_streamsetconnection_t = ctypes.c_void_p
astra_reader_t = ctypes.c_void_p
astra_depthstream_t = ctypes.c_void_p
astra_colorstream_t = ctypes.c_void_p
astra_reader_frame_t = ctypes.c_void_p
astra_depthframe_t = ctypes.c_void_p
astra_colorframe_t = ctypes.c_void_p

class AstraImageMetadata(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelFormat", ctypes.c_int32),
        ("reserved", ctypes.c_uint32)
    ]

class AstraCamera:
    """Astra Pro camera wrapper - RGB + Depth"""
    def __init__(self):
        self.sensor = astra_streamsetconnection_t()
        self.reader = astra_reader_t()
        self.depth_stream = astra_depthstream_t()
        self.color_stream = astra_colorstream_t()
        self.initialized = False
        self.depth_width = 640
        self.depth_height = 480
        self.color_width = 640
        self.color_height = 480
        self.has_color = False
        self.rgb_cap = None
        self.use_opencv_rgb = False

    def initialize(self):
        print("[INFO] Initializing Astra SDK...")
        rc = astra_core.astra_initialize()
        if rc != 0:
            print(f"[ERROR] astra_initialize failed: {rc}")
            return False
        print("[INFO] Astra SDK initialized successfully")

        rc = astra_core.astra_streamset_open(b"device/default", ctypes.byref(self.sensor))
        if rc != 0:
            print(f"[ERROR] astra_streamset_open failed: {rc}")
            astra_core.astra_terminate()
            return False
        print(f"[INFO] Device opened")

        rc = astra_core.astra_reader_create(self.sensor, ctypes.byref(self.reader))
        if rc != 0:
            print(f"[ERROR] astra_reader_create failed: {rc}")
            astra_core.astra_streamset_close(self.sensor)
            astra_core.astra_terminate()
            return False
        print("[INFO] Reader created successfully")

        # Get depth stream
        rc = astra.astra_reader_get_depthstream(self.reader, ctypes.byref(self.depth_stream))
        if rc != 0:
            print(f"[ERROR] astra_reader_get_depthstream failed: {rc}")
            astra_core.astra_reader_destroy(ctypes.byref(self.reader))
            astra_core.astra_streamset_close(self.sensor)
            astra_core.astra_terminate()
            return False
        print("[INFO] Depth stream obtained successfully")

        # Start depth stream
        rc = astra_core.astra_stream_start(self.depth_stream)
        if rc != 0:
            print(f"[ERROR] astra_stream_start (depth) failed: {rc}")
            astra_core.astra_reader_destroy(ctypes.byref(self.reader))
            astra_core.astra_streamset_close(self.sensor)
            astra_core.astra_terminate()
            return False
        print("[INFO] Depth stream started")

        # Attempt to get color stream (Astra SDK method)
        try:
            rc = astra.astra_reader_get_colorstream(self.reader, ctypes.byref(self.color_stream))
            if rc == 0:
                print("[INFO] Color stream obtained successfully (Astra SDK)")
                rc = astra_core.astra_stream_start(self.color_stream)
                if rc == 0:
                    print("[INFO] Color stream started (Astra SDK)")
                    self.has_color = True
                else:
                    print(f"[WARN] Failed to start color stream (Astra SDK): {rc}")
                    self.color_stream = None
            else:
                print(f"[INFO] Color stream not available (Astra SDK): {rc}")
                self.color_stream = None
        except Exception as e:
            print(f"[INFO] Color stream initialization exception (Astra SDK): {e}")
            self.color_stream = None

        # If Astra SDK method fails, try OpenCV UVC method
        if not self.has_color:
            print("[INFO] Trying to open RGB camera via OpenCV...")
            for device_id in [4, 2, 0, 6, 8]:
                self.rgb_cap = cv2.VideoCapture(device_id)
                if self.rgb_cap.isOpened():
                    self.rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.rgb_cap.set(cv2.CAP_PROP_FPS, 30)
                    ret, test_frame = self.rgb_cap.read()
                    if ret and test_frame is not None:
                        print(f"[INFO] RGB camera opened via OpenCV ( /dev/video{device_id})")
                        self.use_opencv_rgb = True
                        self.has_color = True
                        break
                    else:
                        self.rgb_cap.release()
                        self.rgb_cap = None

            if not self.use_opencv_rgb:
                print("[WARN] Could not open RGB camera via OpenCV, depth only will be shown")

        self.initialized = True
        return True

    def read_frames(self):
        """Read RGB and depth frames"""
        if not self.initialized:
            return None, None

        depth_array = None
        color_array = None

        # If using OpenCV to read RGB
        if self.use_opencv_rgb and self.rgb_cap:
            ret, color_array = self.rgb_cap.read()
            if ret and color_array is not None:
                if color_array.shape[:2] != (480, 640):
                    color_array = cv2.resize(color_array, (640, 480))

        # Read depth (and possibly RGB) via Astra SDK
        astra_core.astra_update()
        frame = astra_reader_frame_t()
        rc = astra_core.astra_reader_open_frame(self.reader, 100, ctypes.byref(frame))

        if rc == 0:
            try:
                # Read depth frame
                depth_frame = astra_depthframe_t()
                rc = astra.astra_frame_get_depthframe(frame, ctypes.byref(depth_frame))
                if rc == 0:
                    depth_length = ctypes.c_uint32()
                    astra.astra_depthframe_get_data_byte_length(depth_frame, ctypes.byref(depth_length))
                    if depth_length.value > 0:
                        num_pixels = depth_length.value // 2
                        depth_buffer = (ctypes.c_int16 * num_pixels)()
                        astra.astra_depthframe_copy_data(depth_frame, depth_buffer)
                        metadata = AstraImageMetadata()
                        astra.astra_depthframe_get_metadata(depth_frame, ctypes.byref(metadata))
                        self.depth_width = metadata.width
                        self.depth_height = metadata.height
                        depth_array = np.ctypeslib.as_array(depth_buffer).copy()
                        depth_array = depth_array.reshape((self.depth_height, self.depth_width))

                # If reading RGB via Astra SDK
                if not self.use_opencv_rgb and self.has_color and self.color_stream:
                    color_frame = astra_colorframe_t()
                    rc = astra.astra_frame_get_colorframe(frame, ctypes.byref(color_frame))
                    if rc == 0:
                        color_length = ctypes.c_uint32()
                        astra.astra_colorframe_get_data_byte_length(color_frame, ctypes.byref(color_length))
                        if color_length.value > 0:
                            num_pixels = color_length.value // 3
                            color_buffer = (ctypes.c_uint8 * color_length.value)()
                            astra.astra_colorframe_copy_data(color_frame, color_buffer)
                            color_metadata = AstraImageMetadata()
                            astra.astra_colorframe_get_metadata(color_frame, ctypes.byref(color_metadata))
                            self.color_width = color_metadata.width
                            self.color_height = color_metadata.height
                            color_array = np.ctypeslib.as_array(color_buffer).copy()
                            color_array = color_array.reshape((self.color_height, self.color_width, 3))
                            color_array = cv2.cvtColor(color_array, cv2.COLOR_RGB2BGR)
            finally:
                astra_core.astra_reader_close_frame(ctypes.byref(frame))

        return color_array, depth_array

    def release(self):
        if not self.initialized:
            return
        print("[INFO] Releasing resources...")
        if self.has_color and self.color_stream:
            astra_core.astra_stream_stop(self.color_stream)
        astra_core.astra_stream_stop(self.depth_stream)
        astra_core.astra_reader_destroy(ctypes.byref(self.reader))
        astra_core.astra_streamset_close(self.sensor)
        astra_core.astra_terminate()
        if self.rgb_cap:
            self.rgb_cap.release()
        self.initialized = False
        print("[INFO] Resources released")

def colorize_depth(depth, min_dist=300):
    """Colorize depth map - JET colormap: near=warm colors(red), far=cool colors(blue)"""
    global MAX_DIST, INVALID_COLOR

    if depth is None:
        return None

    valid_mask = depth > 0
    h, w = depth.shape

    # Create output image, fill invalid pixels with specified color
    depth_norm = np.zeros((h, w), dtype=np.uint8)

    if np.any(valid_mask):
        depth_valid = depth[valid_mask].astype(np.float32)

        # Normalize to 0-255: near(min_dist)=255, far(MAX_DIST)=0
        # This maps near objects to red, far objects to blue in JET colormap
        normalized = 255 - np.clip((depth_valid - min_dist) / (MAX_DIST - min_dist) * 255, 0, 255)
        depth_norm[valid_mask] = normalized.astype(np.uint8)

    # Apply JET colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Set invalid regions to specified color
    depth_colored[~valid_mask] = INVALID_COLOR

    return depth_colored

def gray_depth(depth, min_dist=300):
    """Generate grayscale depth map - near=white(255), far=black(0)"""
    global MAX_DIST

    if depth is None:
        return None

    valid_mask = depth > 0
    h, w = depth.shape

    # Default fill with white for invalid depth
    gray = np.full((h, w), 255, dtype=np.uint8)

    if np.any(valid_mask):
        depth_valid = depth[valid_mask].astype(np.float32)
        # Normalize: near=min_dist -> 255(white), far=MAX_DIST -> 0(black)
        gray_values = 255 - np.clip((depth_valid - min_dist) / (MAX_DIST - min_dist) * 255, 0, 255)
        gray[valid_mask] = gray_values.astype(np.uint8)

    # Convert to 3-channel for JPEG encoding
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray_bgr

def get_grid_distances(depth, cols=10, rows=10):
    """Get distances (meters) at grid center points"""
    if depth is None:
        return [0.0] * (cols * rows)

    h, w = depth.shape
    distances = []

    for row in range(rows):
        for col in range(cols):
            cy = int((row + 0.5) * h / rows)
            cx = int((col + 0.5) * w / cols)
            dist_mm = depth[cy, cx]

            if dist_mm > 0:
                dist_m = min(dist_mm / 1000.0, 10.0)  # Limit to max 10 meters
            else:
                dist_m = 0.0

            distances.append(dist_m)

    return distances