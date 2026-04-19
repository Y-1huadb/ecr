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


if __name__ == "__main__":
    runner = STGCNRunner(
        so_path="./libstgcn_hbdnn.so",
        model_path="./stgcn_x5.bin"
    )

    x = np.fromfile("./-_pn5NxJmok.bin", dtype=np.float32).reshape(1, 3, 32, 18)
    y = runner.run(x)

    print("output shape:", y.shape)
    print("top1:", int(np.argmax(y)))
    print("top5:", np.argsort(y)[-5:][::-1])

    runner.close()