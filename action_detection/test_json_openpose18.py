import json
import numpy as np

import ctypes
import numpy as np


class STGCNRunner:
    def __init__(self, so_path="./libstgcn_hbdnn.so", model_path="./stgcn_x5.bin"):
        self.lib = ctypes.CDLL(so_path)

        self.lib.stgcn_create.argtypes = [ctypes.c_char_p]
        self.lib.stgcn_create.restype = ctypes.c_void_p

        self.lib.stgcn_get_input_numel.argtypes = [ctypes.c_void_p]
        self.lib.stgcn_get_input_numel.restype = ctypes.c_int

        self.lib.stgcn_get_output_numel.argtypes = [ctypes.c_void_p]
        self.lib.stgcn_get_output_numel.restype = ctypes.c_int

        self.lib.stgcn_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
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
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y.size
        )
        if ret != 0:
            raise RuntimeError(f"stgcn_run failed, ret={ret}")
        return y

    def close(self):
        if self.handle:
            self.lib.stgcn_destroy(self.handle)
            self.handle = None

def temporal_sample_or_pad_ctv(x_ctv: np.ndarray, window_size: int) -> np.ndarray:
    # x_ctv: [C, T, V]
    _, t, _ = x_ctv.shape
    if t == window_size:
        return x_ctv
    if t > window_size:
        start = (t - window_size) // 2
        return x_ctv[:, start:start + window_size, :]
    pad_num = window_size - t
    pad = np.repeat(x_ctv[:, -1:, :], pad_num, axis=1)
    return np.concatenate([x_ctv, pad], axis=1)


def load_kinetics_skeleton_json(json_path: str, v: int = 18, frame_index_starts_from_one: bool = True):
    """
    读取单个 kinetics-skeleton json，返回 [C, T, V, M]
    C=3: x, y, score
    """
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

    # 和你之前脚本一致的预处理
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    return data_numpy, video_info


def select_person(sample_ctvm: np.ndarray, mode: str = "best"):
    """
    sample_ctvm: [C, T, V, M]
    返回选中的 [C, T, V]
    """
    _, _, _, m = sample_ctvm.shape
    if m == 1:
        return sample_ctvm[..., 0], 0
    if mode == "first":
        return sample_ctvm[..., 0], 0

    # 用 score 总和最大的行人
    scores = sample_ctvm[2].sum(axis=(0, 1))
    idx = int(np.argmax(scores))
    return sample_ctvm[..., idx], idx


def build_stgcn_input_from_json(
    json_path: str,
    window_size: int = 32,
    person_mode: str = "best",
    v: int = 18,
):
    """
    返回:
      inp: [1, C, T, V] float32
      meta: 原始 json 元信息
      chosen_person: 选中的 person id
    """
    sample_ctvm, meta = load_kinetics_skeleton_json(json_path, v=v)
    ctv, chosen_person = select_person(sample_ctvm, mode=person_mode)
    ctv = temporal_sample_or_pad_ctv(ctv, window_size)
    inp = ctv[None, ...].astype(np.float32)   # [1,C,T,V]
    inp = np.ascontiguousarray(inp)
    return inp, meta, chosen_person



json_path = "/home/sunrise/Desktop/data/-_pn5NxJmok.json"

runner = STGCNRunner(
    so_path="./libstgcn_hbdnn.so",
    model_path="/home/sunrise/Desktop/ECR/models/stgcn_x5.bin"
)

x, meta, chosen_person = build_stgcn_input_from_json(
    json_path=json_path,
    window_size=32,
    person_mode="best",
    v=18
)

print("input shape:", x.shape)            # (1, 3, 32, 18)
print("input dtype:", x.dtype)            # float32
print("chosen_person:", chosen_person)

logits = runner.run(x)

print("logits shape:", logits.shape)      # 一般是 (400,)
top1 = int(np.argmax(logits))
top5 = np.argsort(logits)[-5:][::-1]

print("top1 index:", top1)
print("top5 index:", top5)
print("top5 score:", logits[top5])

runner.close()