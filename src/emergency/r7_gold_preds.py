"""R7 (heliosli bundle 8c7b666e, 4 fold models, `heliosli/rsna-knee-compact-r7cnxfsss`) on gold_58, run locally with the
same worker and arguments as r6_gold_preds.py (R6's v4c compact_infer.py, sha 8401812c, which R7's manifest also pins):
the bundle manifest's arguments + flip TTA + the bundle's own fluid and state stackers. --dump-logits keeps every fold
model's plain and mirrored logits, so fold mixes can be formed offline without another run.

Mission 2026-10-21 (Track 1): R7's per-finding gold_58 AUC next to R6 (0.9367), s6fall, s6ft, S6 and Raptor.
R7 trained with a fastMRI auxiliary task (manifest run_config.fm_aux): final pick 1 only, never final-strict.

Usage: python -I src/emergency/r7_gold_preds.py [--limit N]
Writes output/emergency_research/task1/r7/ (worker output: predictions.csv, logits.json, stacker dumps, receipt).
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
BUNDLE = f"{PW}/rsna-knee-compact-r7cnxfsss"
MANIFEST_SHA = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"   # as downloaded 2026-10-21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32, help="encoder windows per step (worker default 192 OOMs on 6 GB)")
    ap.add_argument("--out", default=f"{R}/output/emergency_research/task1/r7")
    a = ap.parse_args()
    man = Path(BUNDLE, "manifest.json")
    assert hashlib.sha256(man.read_bytes()).hexdigest() == MANIFEST_SHA, "R7 manifest differs from the downloaded one"
    m = json.loads(man.read_text())
    for name in m["models"] + [m["state_spec"], m["side_model"], m["fluid_stacker"], m["state_stacker"]]:
        assert hashlib.sha256(Path(BUNDLE, name).read_bytes()).hexdigest() == m["sha256"][name], name
    assert hashlib.md5(Path(SRC, "compact_train.py").read_bytes()).hexdigest() == m["trainer_md5"], "trainer differs from the bundle's"
    assert hashlib.sha256(Path(SRC, "compact_infer.py").read_bytes()).hexdigest() == m["src_sha256"]["compact_infer.py"], "worker differs"
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
            "--anchor-weights", ANATOMY,
            "--dump-logits", str(out / "logits.json")]
    argv = ["compact_infer.py", "--data", f"{R}/data/Raw_dicom", "--split", "train", "--code", CODE, "--models", BUNDLE,
            "--arm", m["arm"], "--assignment", str(assign), "--output", str(out), "--tmp", "D:/rsna_work/tmp_task1/r7",
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
