"""evaluate_specialist_gold.py: Evaluates any compact specialist model on Gold-58.

Implements the NEW MANDATORY ADOPTION RULE (overriding old CI gates):
A trained model / specialist head is ADOPTED if ANY of the following hold:
1. Standalone Gold-58 Macro AUC > baseline (s6fall 0.9314).
2. Point-estimate ensemble delta AUC > +0.0002 on Gold-58.
3. Group mean AUC on any of the 3 clinical abnormality groups exceeds s6fall by >= +0.0020.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]

GROUPS = {
    "Group 1 (Ligaments & Menisci)": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "Group 2 (Cartilage & Bone)": ["Medial OA", "Lateral OA", "PF OA", "Fracture", "Contusion"],
    "Group 3 (Fluid & Synovium)": ["Effusion", "Baker's", "Synovitis"],
}

def sig(z): return 1.0 / (1.0 + np.exp(-z))

def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def macro(Y, S):
    return float(np.nanmean([auc(Y[:, j], S[:, j]) for j in range(Y.shape[1])]))

def load(folder):
    out = {}
    for f in glob.glob(f"{T1}/{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["uid"])] = {k: z[k] for k in z.files if k != "uid"}
    return out

def pct(x):
    r = rankdata(x, method="average")
    return (r - 1.0) / max(1, len(r) - 1)

def evaluate_checkpoint(ckpt_path, out_pred_dir=None):
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, "D:/rsna_work/public_code/s6_src")
    import compact_train as CT

    # Load model
    m = CT.Net("convnext_small", None, freeze_blocks=16, crop=288, state_spec=None, in_chans=3, spatial=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    m = m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)

    # Gold studies
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    gold_ids = sorted(gold.index)

    # Score each gold study from cache
    cache_dir = "D:/rsna_work/s6ft_cache_all/studies"
    preds = {}
    with torch.inference_mode():
        for uid in gold_ids:
            f = os.path.join(cache_dir, f"{uid}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            img, meta = z["jpg"], z["meta"]
            # decode windows
            import cv2
            k = len(meta)
            off = z["off"]
            w_imgs = np.empty((k, 3, 320, 320), np.uint8)
            for w in range(k):
                for ch in range(3):
                    q = 3 * w + ch
                    w_imgs[w, ch] = cv2.imdecode(img[off[q]:off[q + 1]], cv2.IMREAD_GRAYSCALE)
            # Center crop to 288
            x = torch.from_numpy(w_imgs).to(dev)[None, :, :, 16:304, 16:304]
            mt = torch.from_numpy(meta.astype(np.float32)).to(dev)[None]
            mask = torch.ones((1, k), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = m(x, mt, mask, chunk=16).float()[0]
            prob = torch.sigmoid(logit.double()).cpu().numpy()
            preds[uid] = prob

    ids = [u for u in gold_ids if u in preds]
    assert len(ids) == 58, f"Only {len(ids)} gold studies scored"
    Y = gold.loc[ids, LAB].values.astype(int)
    P_cand = np.stack([preds[u] for u in ids])

    cand_macro = macro(Y, P_cand)
    s6f_data = load("s6fall")
    P_s6fall = sig(np.stack([s6f_data[u]["z_plain"][0] for u in ids]))
    s6fall_macro = macro(Y, P_s6fall)

    print(f"\n=== EVALUATION REPORT for {os.path.basename(ckpt_path)} ===")
    print(f"Standalone Gold-58 Macro AUC: {cand_macro:.4f} (vs s6fall {s6fall_macro:.4f}, diff: {cand_macro - s6fall_macro:+.4f})")

    # Group evaluation
    adopted_reasons = []
    if cand_macro > s6fall_macro:
        adopted_reasons.append(f"Standalone Macro AUC {cand_macro:.4f} > baseline {s6fall_macro:.4f} (+{cand_macro - s6fall_macro:.4f})")

    for gname, glabels in GROUPS.items():
        g_idx = [LAB.index(c) for c in glabels]
        cand_g = np.mean([auc(Y[:, j], P_cand[:, j]) for j in g_idx])
        base_g = np.mean([auc(Y[:, j], P_s6fall[:, j]) for j in g_idx])
        diff_g = cand_g - base_g
        print(f"  {gname:35s}: cand {cand_g:.4f} vs base {base_g:.4f} ({diff_g:+.4f})")
        if diff_g >= 0.0020:
            adopted_reasons.append(f"Group {gname} delta {diff_g:+.4f} >= +0.0020")

    # Per-finding table
    print("\nPer-finding AUC comparison:")
    for j, c in enumerate(LAB):
        a_cand = auc(Y[:, j], P_cand[:, j])
        a_base = auc(Y[:, j], P_s6fall[:, j])
        print(f"  {c:18s}: {a_cand:.4f} vs base {a_base:.4f} ({a_cand - a_base:+.4f})")

    print("\n=== ADOPTION DECISION ===")
    if adopted_reasons:
        print("STATUS: ADOPTED!")
        for r in adopted_reasons:
            print(f"  [PASS] {r}")
    else:
        print("STATUS: NOT ADOPTED (Does not meet new adoption threshold)")

    return cand_macro, adopted_reasons

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "D:/rsna_work/s6ft_v2_local/s6ft_ema.pt"
    evaluate_checkpoint(ckpt)
