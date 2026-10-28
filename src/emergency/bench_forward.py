"""GPU forward cost per study for encoder / input-size / window-count settings (efficiency analysis).

  python src/emergency/bench_forward.py   (run when the GPU is otherwise idle)
Prints ms per study (3 slots, fp16/bf16 autocast, batch of 1-4 studies) and peak memory; writes
output/emergency_research/bench_forward.json.
"""
import json
import sys
import time

import torch

R = "D:/Coding/kaggle/RSNA"
sys.path.insert(0, f"{R}/src/imaging")
from model import KneeMIL  # noqa: E402

SETTINGS = [  # (backbone, size, windows per slot, studies per batch)
    ("convnext_tiny.fb_in22k_ft_in1k", 224, 8, 4),
    ("convnext_tiny.fb_in22k_ft_in1k", 224, 4, 4),
    ("convnext_tiny.fb_in22k_ft_in1k", 320, 12, 1),
    ("convnext_tiny.fb_in22k_ft_in1k", 320, 6, 2),
    ("convnext_tiny.fb_in22k_ft_in1k", 320, 34, 1),       # about every slice of a typical series
    ("convnext_nano.in12k_ft_in1k", 224, 8, 4),
    ("efficientnet_b0.ra_in1k", 224, 8, 4),
]


@torch.no_grad()
def time_setting(bb, size, k, bs, dt, n=12):
    m = KneeMIL(bb, pretrained=False, n_slots=6, agg="attn12").cuda().eval()
    x = torch.rand(bs, 3, k, 3, size, size, device="cuda")
    mask, pos = torch.ones(bs, 3, device="cuda"), torch.rand(bs, 3, k, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        with torch.autocast("cuda", dtype=dt):
            m(x, mask, pos)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        with torch.autocast("cuda", dtype=dt):
            m(x, mask, pos)
    torch.cuda.synchronize()
    ms = (time.time() - t) / n / bs * 1000
    return round(ms, 1), round(torch.cuda.max_memory_allocated() / 1e9, 2)


def main():
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    out = {"gpu": torch.cuda.get_device_name(0), "dtype": str(dt), "results": []}
    for bb, size, k, bs in SETTINGS:
        try:
            ms, mem = time_setting(bb, size, k, bs, dt)
        except Exception as ex:
            ms, mem = None, f"{type(ex).__name__}"
        r = {"backbone": bb, "size": size, "windows_per_slot": k, "batch_studies": bs, "ms_per_study": ms, "peak_gb": mem}
        out["results"].append(r)
        print(r, flush=True)
        torch.cuda.empty_cache()
    json.dump(out, open(f"{R}/output/emergency_research/bench_forward.json", "w"), indent=1)


if __name__ == "__main__":
    main()
