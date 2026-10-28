"""Final artifact package (no competition data): reports, code, candidate notebooks, experiment tables, plots.

Allowlist only. Refused outright: DICOMs, arrays, checkpoints, any CSV with a StudyInstanceUID column, and any file that
still contains a competition study UID after redaction (Markdown copies have UIDs replaced by <study-uid>).
Checkpoints are listed (path + SHA-256 from experiments.csv) in CHECKPOINTS.md, not copied.

Usage: python -I src/emergency/package_artifacts.py [out_dir]   (default D:/rsna_work/final_artifacts)
"""
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile

import pandas as pd

R = "D:/Coding/kaggle/RSNA"
ALLOW = [
    "reports/current_model_mathematical_deep_dive.md",
    "reports/competition_and_medical_imaging_research.md",
    "reports/emergency_initial_assessment.md",
    "reports/emergency_experiment_plan.md",
    "reports/emergency_experiment_results.md",
    "reports/efficiency_prize_strategy.md",
    "reports/blackwell_gpu_plan.md",
    "reports/FINAL_EMERGENCY_COMPETITION_REPORT.md",
    "reports/competition_rules_audit.md",
    "reports/repo_remediation_plan.md",
    "reports/gap_discovery_audit.md",
    "reports/eda/emergency/pareto_accuracy_runtime.png",
    "output/emergency_research/task1/task2_gold_eval.json",          # plan 12: aggregate metrics only, no study ids
    "output/emergency_research/task1/task2b_oof_mirror_check.json",
    "output/emergency_research/task1/task3_adapter.json",
    "output/emergency_research/task1/task1b_oof_memorisation.json",
    "output/emergency_research/experiments.csv",
    "output/emergency_research/comparisons.csv",
    "output/emergency_research/pareto_points.csv",
    "output/emergency_research/experiment_notes.json",
    "output/emergency_research/bench_forward.json",
    "output/emergency_research/bench_preprocess_50_4w.json",
    "output/emergency_research/gold_look_*.json",
    "output/emergency_research/queue_oof.json",
    "notebooks/EMERGENCY_SUBMISSION_GUIDE.md",
    "notebooks/rsna_knee_image_model_inference_emergency.ipynb",
    "notebooks/rsna_knee_infer_standalone.py",
    "notebooks/part_c_ours_blend_cell.py",
    "notebooks/candidate_A/*/*",
    "src/emergency/*.py",
    "src/imaging/*.py",
    "src/labeling/metrics.py",
    "configs/*",
]
UID = re.compile(r"1\.2\.826\.0\.1\.3680043(\.\d+)+")
REFUSE_EXT = {".dcm", ".npz", ".npy", ".pt", ".pth", ".safetensors", ".parquet", ".pkl"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else "D:/rsna_work/final_artifacts"
    stamp = time.strftime("%Y%m%d_%H%M")
    stage = os.path.join(out_root, f"rsna_knee_artifacts_{stamp}")
    os.makedirs(stage, exist_ok=False)
    files = sorted({f for pat in ALLOW for f in glob.glob(os.path.join(R, pat)) if os.path.isfile(f)})
    manifest, redacted = [], []
    for src in files:
        rel = os.path.relpath(src, R).replace("\\", "/")
        ext = os.path.splitext(src)[1].lower()
        assert ext not in REFUSE_EXT, f"refused file type: {rel}"
        if ext == ".csv":
            cols = pd.read_csv(src, nrows=0).columns
            assert "StudyInstanceUID" not in cols, f"refused: per-study table {rel}"
        dst = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if ext in (".md", ".txt"):
            text = open(src, encoding="utf-8").read()
            new = UID.sub("<study-uid>", text)
            if new != text:
                redacted.append(rel)
            open(dst, "w", encoding="utf-8").write(new)
        else:
            shutil.copyfile(src, dst)
        if ext not in (".png",):
            content = open(dst, encoding="utf-8", errors="replace").read()
            assert not UID.search(content), f"refused: a study UID remains in {rel}"
        manifest.append({"path": rel, "bytes": os.path.getsize(dst), "sha256": sha256(dst)})

    ex = pd.read_csv(os.path.join(R, "output/emergency_research/experiments.csv"))
    ck = ex[ex["checkpoint_sha256"].notna()][["id", "run_id", "checkpoint", "checkpoint_sha256"]]
    lines = ["# Checkpoints (listed, not included)", "",
             "Weights trained on competition data stay out of this package. They are in the owner's private Kaggle "
             "datasets (`rsna-knee-image-model`, `rsna-knee-v3-s1`) and on the local disk.", "",
             "| Experiment | Run | Local path | SHA-256 |", "|---|---|---|---|"]
    lines += [f"| {r.id} | {r.run_id} | `{r.checkpoint}` | `{r.checkpoint_sha256}` |" for r in ck.itertuples()]
    open(os.path.join(stage, "CHECKPOINTS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"created": stamp, "files": manifest, "redacted_study_uids_in": redacted,
               "excluded_by_design": ["data/ (labels, pseudo-labels, study index: derived from competition data)",
                                      "DICOMs, report text, caches, prediction sets with study UIDs, checkpoints"]},
              open(os.path.join(stage, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    zpath = stage + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(stage):
            for f in fs:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, out_root))
    print(f"{len(manifest)} files, {len(ck)} checkpoints listed, UIDs redacted in {redacted}")
    print("wrote", zpath, os.path.getsize(zpath) // 1024, "KB")


if __name__ == "__main__":
    main()
