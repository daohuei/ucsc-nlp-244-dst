import subprocess
import sys
import os

import torch
import pandas as pd


if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode("utf-8")),
        names=["memory.used", "memory.free"],
        skiprows=1,
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(
        lambda x: int(x.rstrip(" MiB"))
    )
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Returning GPU{} with {} free MiB".format(
            idx, gpu_df.iloc[idx]["memory.free"]
        )
    )
    return idx


def choose_free_device():
    free_gpu_id = 0
    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        print(f"using GPU id: {free_gpu_id}")
        # torch.cuda.set_device(0)

    return free_gpu_id


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    return device, choose_free_device()


if __name__ == "__main__":
    free_gpu_id = get_free_gpu()
    print(free_gpu_id)
