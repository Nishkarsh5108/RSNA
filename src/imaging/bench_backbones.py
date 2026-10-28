"""Benchmark candidate 2D backbones for 2.5D fine-tuning on the local GPU.

Measures training images/s (fwd+bwd+AdamW, bf16 autocast) and peak VRAM at 224 px
for a given number of images per step, with and without gradient checkpointing.
"""
import os
import sys
import time

import torch
import timm

os.environ.setdefault("HF_HOME", "D:/rsna_work/hf_cache")
CANDS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "convnext_nano.in12k_ft_in1k", "convnext_tiny.fb_in22k_ft_in1k", "tf_efficientnetv2_s.in21k_ft_in1k",
    "efficientnet_b0.ra_in1k", "resnet34.a1_in1k", "coatnet_0_rw_224.sw_in1k"]
N_IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 48
dev = "cuda"
for name in CANDS:
    for ckpt in (False, True):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = timm.create_model(name, pretrained=True, num_classes=0).to(dev).to(memory_format=torch.channels_last)
            if ckpt:
                m.set_grad_checkpointing(True)
            head = torch.nn.Linear(m.num_features, 12).to(dev)
            opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-4)
            x = torch.randn(N_IMG, 3, 224, 224, device=dev).to(memory_format=torch.channels_last)
            for i in range(8):
                if i == 3:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = head(m(x)).float().pow(2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); dt = (time.time() - t0) / 5
            print(f"{name:38s} ckpt={ckpt!s:5s} params={sum(p.numel() for p in m.parameters())/1e6:5.1f}M "
                  f"{N_IMG/dt:7.1f} img/s  step {dt:.3f}s  peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
            del m, head, opt, x, loss
        except Exception as ex:
            print(f"{name:38s} ckpt={ckpt} FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            torch.cuda.empty_cache()
