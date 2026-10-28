"""Where does the per-finding attention (attn12 / tfm) look? Mean attention mass per (slot, window
position) for each finding over pseudo-val studies, plus the attention-weighted mean slice position
per finding and plane. After laterality normalisation, sagittal position 0 = lateral, 1 = medial for
every knee, so medial vs lateral targets should separate along the sagittal axis if the model uses
the anatomy (a check of the orientation fix). Slice normals (uniform in this data): coronal +y, so
position 0 -> 1 runs anterior -> posterior; axial +z, so inferior -> superior. Writes reports/eda/imaging/attention_<run>.png/.csv.
Usage: python attention_map.py RUN_ID [N_STUDIES]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import KneeStudyDataset  # noqa: E402
from model import KneeMIL  # noqa: E402
from study_index import SLOT_NAMES, TARGETS  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


@torch.no_grad()
def attention(m, x01, mask, pos):
    """Mirror of KneeMIL.forward up to the per-finding attention -> [B, S, K, 12] weights."""
    B, S, K = x01.shape[:3]
    valid = mask > 0
    feats = x01.new_zeros(B, S, K, m.proj[1].out_features)
    xv = x01[valid]
    f = m.encode(xv.flatten(0, 1).contiguous(memory_format=torch.channels_last))
    feats[valid] = m.proj(f.float()).view(xv.shape[0], K, -1).to(feats.dtype)
    tok = feats + m.slot_emb.weight[:S].view(1, S, 1, -1) + m.pos_mlp(pos.unsqueeze(-1))
    tok = tok.flatten(1, 2)
    tmask = valid.unsqueeze(-1).expand(B, S, K).flatten(1)
    if m.agg == "tfm":
        tok = m.tfm(tok, src_key_padding_mask=~tmask)
    a = torch.softmax(m.att(tok).masked_fill(~tmask.unsqueeze(-1), -1e4), 1)
    return a.view(B, S, K, -1)


def main():
    run = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    ck = torch.load(f"{R}/output/imaging_v5/runs/{run}/ckpt_best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["agg"] in ("attn12", "tfm"), "needs a per-finding attention model"
    m = KneeMIL(a["backbone"], pretrained=False, n_slots=6, agg=a["agg"], use_pos=not a.get("no_pos", False))
    m.load_state_dict(ck["model"]); m.eval().cuda()
    idx = pd.read_csv(a["index"])
    va = idx[idx.fold == a["val_fold"]].head(n).reset_index(drop=True)
    slots = [int(s) for s in str(a["slots"]).split(",")]
    ds = KneeStudyDataset(va, a["cache"], slots, a["K"], train=False, lo=a["lo"], hi=a["hi"], orient=not a.get("no_orient", False),
                              sag_pos_fix=a.get("sag_pos_fix", False))
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    S, K = len(slots), a["K"]
    acc, cnt = np.zeros((S, K, 12)), 0
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            att = attention(m, b["x"].cuda().float().div_(255), b["mask"].cuda(), b["pos"].cuda()).float().cpu().numpy()
        acc += att.sum(0); cnt += att.shape[0]
    acc /= cnt
    # window k sits at anatomical position linspace(lo, hi, K)[k] for BOTH knees (right-knee sagittal windows are
    # mirrored), so the position comes from the window index, not from the model's position input
    kpos = np.linspace(a["lo"], a["hi"], K)
    mean_pos = (acc * kpos[None, :, None]).sum(1) / np.maximum(acc.sum(1), 1e-9)
    names = [SLOT_NAMES[s] for s in slots]
    rows = []
    for j, t in enumerate(TARGETS):
        r = {"target": t}
        for si, nm in enumerate(names):
            r[f"mass_{nm}"] = acc[si, :, j].sum()
            r[f"meanpos_{nm}"] = mean_pos[si, j]
        rows.append(r)
    out = pd.DataFrame(rows)
    os.makedirs(f"{R}/reports/eda/imaging", exist_ok=True)
    out.to_csv(f"{R}/reports/eda/imaging/attention_{run}.csv", index=False)
    fig, axs = plt.subplots(3, 4, figsize=(16, 8))
    for j, (t, ax) in enumerate(zip(TARGETS, axs.flat)):
        im = ax.imshow(acc[:, :, j], aspect="auto", cmap="viridis", vmin=0)
        ax.set_yticks(range(S)); ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(range(K)); ax.set_xticklabels([f"{p:.2f}" for p in np.linspace(a["lo"], a["hi"], K)], fontsize=6)
        ax.set_title(t, fontsize=10); plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"{run}: mean per-finding attention by series and window position (n={cnt} val studies; "
                 f"position 0->1: sagittal lateral->medial, coronal anterior->posterior, axial inferior->superior)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{R}/reports/eda/imaging/attention_{run}.png", dpi=100); plt.close()
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
