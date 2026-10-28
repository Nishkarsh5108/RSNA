"""R6 (heliosli bundle fe055406, 4 fold models) on gold_58, run locally with R6's own worker, exactly as goodpjw2008's v27
notebook calls it (Part A2): compact_infer.py v4c with the bundle manifest's arguments + flip TTA + the fluid (Effusion)
and state (Baker's) stackers. Only differences: local DICOMs (split train), no deadline, eager (v27's --compile is
checked against eager inside the worker), fp16 autocast as on the T4, 32 windows per encoder step, and this process capped at 60 % of the 6 GB card
(Windows spills to system RAM otherwise; see task1_compact_preds.py).

Owner's brief 2026-10-20 (final-max, Track A): gold_58 scores for R6 so the final-max blend can be checked on gold.
R6 is the fastMRI-assisted leg; the owner approved it for final pick 1 only.

Usage: python -I src/emergency/r6_gold_preds.py [--limit N]
Writes output/emergency_research/task1/r6/ (worker output: predictions.csv, stacker dumps, log).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
SRC = "C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6"
CODE = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/cloud_code"
ANATOMY = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt"
BUNDLE = f"{PW}/rsna-knee-r6-fe055406"
MANIFEST_SHA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"   # the hash v27 pins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32, help="encoder windows per step (worker default 192 OOMs on 6 GB)")
    ap.add_argument("--out", default=f"{R}/output/emergency_research/task1/r6")
    a = ap.parse_args()
    man = Path(BUNDLE, "manifest.json")
    assert hashlib.sha256(man.read_bytes()).hexdigest() == MANIFEST_SHA, "R6 manifest differs from the one v27 pins"
    m = json.loads(man.read_text())
    assert hashlib.md5(Path(SRC, "compact_train.py").read_bytes()).hexdigest() == m["trainer_md5"], "trainer differs from the bundle's"
    assert hashlib.sha256(Path(ANATOMY).read_bytes()).hexdigest() == m["anchor_weights_sha256"], "anatomy.pt differs"
    import pandas as pd
    st = pd.read_csv(f"{R}/output/emergency_research/task1/task1_studies.csv", dtype={"StudyInstanceUID": str})
    uids = st.loc[st["set"] == "gold_58", "StudyInstanceUID"].tolist()
    assert len(uids) == 58, len(uids)
    if a.limit:
        uids = uids[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assign = out / "assignment.json"
    assign.write_text(json.dumps(uids))
    args = ["--backbone", m["backbone"], "--size", str(m["size"]), "--crop", str(m["crop"]),
            "--eval-stride", str(m.get("eval_stride", 3)), "--views", m.get("views", "v1"),
            "--state-spec", f"{BUNDLE}/{m['state_spec']}",
            "--side-model", f"{BUNDLE}/{m['side_model']}", "--side-threshold", str(m.get("side_threshold", 2.0)),
            "--side-position-mm", str(m.get("side_position_mm", 20.0)), "--side-max-x", str(m.get("side_max_x", 0.0)),
            "--slices", str(m.get("slices", 3)), "--phys-mm", str(m.get("phys_mm", 0.0)),
            "--trainer-md5", m["trainer_md5"],
            "--fluid-stacker", f"{BUNDLE}/{m['fluid_stacker']}",
            "--flip-tta",
            "--state-stacker", f"{BUNDLE}/{m['state_stacker']}",
            "--anchor-weights", ANATOMY]
    argv = ["compact_infer.py", "--data", f"{R}/data/Raw_dicom", "--split", "train", "--code", CODE, "--models", BUNDLE,
            "--arm", m["arm"], "--assignment", str(assign), "--output", str(out), "--tmp", "D:/rsna_work/tmp_task1/r6",
            "--amp", "fp16", "--workers", str(a.workers), "--chunk", str(a.chunk)] + args
    (out / "argv.json").write_text(json.dumps(argv, indent=1))
    os.environ["PYTHONPATH"] = os.pathsep.join([SRC, CODE, os.environ.get("PYTHONPATH", "")])   # spawned loader workers
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [SRC, CODE]
    import torch
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(0.6)
    # imported as a module, not run as __main__: Windows spawns the loader workers, which must import its classes by name
    import compact_infer
    sys.argv = argv
    compact_infer.main()


if __name__ == "__main__":
    main()
