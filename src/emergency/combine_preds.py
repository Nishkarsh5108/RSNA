"""Weighted average of saved fold-0 prediction sets (output/emergency_research/preds/*.npz) -> a new set.

  python src/emergency/combine_preds.py OUT_NAME NAME_A[:w] NAME_B[:w] ...   (probability mean; rows aligned by uid)
"""
import sys

import numpy as np

R = "D:/Coding/kaggle/RSNA"
P = f"{R}/output/emergency_research/preds"


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    ps, ws, uids, wins = [], [], None, 0.0
    for sp in specs:
        name, w = (sp.rsplit(":", 1) + ["1"])[:2] if ":" in sp else (sp, "1")
        z = np.load(f"{P}/{name}.npz", allow_pickle=True)
        order = np.argsort(z["uids"])
        u, p = z["uids"][order], z["p"][order]
        if uids is None:
            uids = u
        assert (u == uids).all(), f"{name}: different studies"
        ps.append(p); ws.append(float(w)); wins += float(z["windows_per_study"])
    ws = np.array(ws) / np.sum(ws)
    p = np.tensordot(ws, np.stack(ps), axes=1)
    np.savez(f"{P}/{out}.npz", uids=uids, p=p, windows_per_study=wins, sec=0.0, parts=np.array(specs))
    print("wrote", out, "from", specs, "weights", ws.round(3).tolist())


if __name__ == "__main__":
    main()
