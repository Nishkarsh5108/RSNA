# Part C: our 2.5D ConvNeXt ensemble, rank-blended into a public notebook's final submission (P1, experiment plan §10).
#
# Paste as the LAST cell of a forked public notebook, after it has written /kaggle/working/submission.csv.
# Attach our release dataset (folder with rsna_knee_infer_standalone.py, knee_*.pt and wheels/).
#
# Per finding: final = rank(0.8 x rank(base) + 0.2 x rank(ours)). The 0.2 share was fixed before any result (§10/10a).
# Our leg runs as its own process with its own packages (installed into /tmp/knee_site), so the base environment is
# unchanged. It runs only if enough time and disk are left. Any skip, failure or timeout keeps the base submission
# byte-for-byte and prints PART_C_SKIPPED / PART_C_FAILED.
import glob, json, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd

OURS_W = 0.2                 # fixed before any result (plan §10); never tuned on the leaderboard
OURS_TIMEOUT_MIN = 75        # hard limit for our leg (local estimate 40-60 min on 1,300 studies; Kaggle unmeasured)
KAGGLE_LIMIT_H = 9.0         # competition limit for the whole notebook
MARGIN_MIN = 15              # kept free after our leg
MIN_FREE_GB = 30             # our cache needs about 20 MB per study (about 25 GB for 1,300 studies)
INPUT_ROOT = os.environ.get("PART_C_INPUT", "/kaggle/input")      # overridable for local tests only
WORK = os.environ.get("PART_C_WORK", "/kaggle/working")
TMP = os.environ.get("PART_C_TMP", "/tmp")
COMP_DIR = os.environ.get("PART_C_COMP")                           # local tests only; on Kaggle the script finds it


def _notebook_t0():
    t0 = globals().get("NB_T0")                     # set by the public bases at their first stage
    if t0 is None:
        try:
            import psutil
            t0 = psutil.Process(os.getpid()).create_time()   # the kernel process starts with the notebook
        except Exception:
            t0 = None
    return t0


def part_c():
    pub, bak = os.path.join(WORK, "submission.csv"), os.path.join(WORK, "_base_submission.csv")
    shutil.copyfile(pub, bak)
    try:
        t0 = _notebook_t0()
        if t0 is None:
            print("PART_C_SKIPPED notebook start time unknown", flush=True); return
        left = KAGGLE_LIMIT_H * 3600 - MARGIN_MIN * 60 - (time.time() - t0)
        free_gb = shutil.disk_usage(TMP).free / 1e9
        if left < OURS_TIMEOUT_MIN * 60:
            print(f"PART_C_SKIPPED only {left / 60:.0f} min left", flush=True); return
        if free_gb < MIN_FREE_GB:
            print(f"PART_C_SKIPPED only {free_gb:.0f} GB free in {TMP}", flush=True); return
        scripts = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "rsna_knee_infer_standalone.py"), recursive=True))
        assert scripts, "our release dataset is not attached"
        model_dir = os.path.dirname(scripts[0])
        assert glob.glob(os.path.join(model_dir, "knee_*.pt")), "no knee_*.pt next to the script"
        env = dict(os.environ)
        try:                                        # the GPU with the most free memory
            import torch
            free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(free)), key=lambda i: free[i]))
        except Exception as e:
            print("PART_C GPU pick skipped:", repr(e), flush=True)
        out = os.path.join(WORK, "submission_ours.csv")
        t = time.time()
        cmd = [sys.executable, "-u", scripts[0], "--model-dir", model_dir, "--out", out,
               "--cache-dir", os.path.join(TMP, "knee_cache")] + (["--comp-dir", COMP_DIR] if COMP_DIR else [])
        rc = subprocess.run(cmd, env=env, timeout=min(left, OURS_TIMEOUT_MIN * 60)).returncode
        assert rc == 0, f"our leg exited with code {rc}"
        base = pd.read_csv(bak, dtype={"StudyInstanceUID": str})
        ours = pd.read_csv(out, dtype={"StudyInstanceUID": str})
        labels = [c for c in base.columns if c != "StudyInstanceUID"]
        assert sorted(ours["StudyInstanceUID"]) == sorted(base["StudyInstanceUID"]), "study IDs differ"
        assert set(labels) <= set(ours.columns), "label columns differ"
        ours = ours.set_index("StudyInstanceUID").loc[base["StudyInstanceUID"], labels].reset_index()
        assert ours[labels].notna().all().all(), "our predictions contain NaN"
        n = len(base)
        if n >= 3:
            assert (ours[labels].nunique() > 1).all(), "our predictions are constant in a column"
        rb = base[labels].rank(method="average", pct=True)
        ro = ours[labels].rank(method="average", pct=True)
        final = base.copy()
        final[labels] = ((1 - OURS_W) * rb + OURS_W * ro).rank(method="average", pct=True).to_numpy()
        assert final["StudyInstanceUID"].tolist() == base["StudyInstanceUID"].tolist()
        assert np.isfinite(final[labels].to_numpy()).all()
        assert ((final[labels].to_numpy() >= 0) & (final[labels].to_numpy() <= 1)).all()
        final.to_csv(pub, index=False)
        rho = {c: round(float(rb[c].corr(ro[c], method="spearman")), 3) for c in labels} if n >= 3 else {}
        print("PART_C_DONE", json.dumps({"weight": OURS_W, "rows": n, "seconds": round(time.time() - t),
                                         "notebook_hours": round((time.time() - t0) / 3600, 2),
                                         "spearman_base_vs_ours": rho}), flush=True)
    except Exception as e:
        shutil.copyfile(bak, pub)
        print("PART_C_FAILED base submission kept:", repr(e), flush=True)


part_c()
