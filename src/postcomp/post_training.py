"""After the fold-2 and fold-3 training runs (friend's account) finish: fetch their weights, add them to the private
weights dataset on the owner's account, build the final ensemble notebooks and hand them to kaggle_flow.py
(push on the owner's account, wait, late-submit once each).

Pre-registered before any result of these two runs exists:
  primary final ensemble  = equal-weight mean of all six models  (A0, A1, A2 = fold 0 with 1st / 7th / fused labels; C1, C2, C3 = folds 1-3, fused)
  ablation                = equal-weight mean of the four fused-label models (A2, C1, C2, C3)
Neither is picked by its leaderboard score; both are reported.

One process, no sub-shells. Tokens are read by kaggle_flow.cli and never printed.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

from kaggle_flow import RUNS, cli, keep_awake

FRIEND = "C:/Users/nishk/.kaggle/kaggle-api_friend.txt"
OWNER = "C:/Users/nishk/.kaggle/kaggle API.txt"
DS_DIR = f"{RUNS}/knee-bp-weights-ds"
DS = "nishkarshsinghal/knee-bp-weights"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = "D:/rsna_work/post_training.log"


def say(*m):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in m)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def status(slug):
    m = re.search(r"KernelWorkerStatus\.(\w+)", cli(FRIEND, "kernels", "status", f"anujchaubey/{slug}"))
    return m.group(1) if m else "?"


def main():
    trained = {"knee-bp-c-f2": ("fold2_ema.pth", "c2_f2.pth"), "knee-bp-c-f3": ("fold3_ema.pth", "c3_f3.pth")}
    # A fold still queued or running at the deadline is left out: the finals must be pushed in time for the owner's
    # presentation (queue waits of 3+ hours per notebook). The primary is then the mean of the models that exist.
    deadline = time.mktime(time.strptime(os.environ.get("TRAIN_DEADLINE", "2026-11-16 05:00"), "%Y-%m-%d %H:%M"))
    keep_awake()
    while True:
        st = {s: status(s) for s in trained}
        say("training status", st)
        if all(v not in ("RUNNING", "QUEUED", "?") for v in st.values()):
            break
        if time.time() > deadline:
            say("deadline reached; continuing with the folds that are complete:", st)
            break
        time.sleep(300)
    trained = {s: v for s, v in trained.items() if st[s] == "COMPLETE"}
    if len(trained) < 2:
        say("not all training runs completed:", st)
    if not trained:
        say("no new fold available; nothing downloaded or submitted")
        sys.exit(1)

    for slug, (src, dst) in trained.items():
        out = f"{RUNS}/{slug}/out"
        os.makedirs(out, exist_ok=True)
        cli(FRIEND, "kernels", "output", f"anujchaubey/{slug}", "-p", out)
        p = f"{out}/{src}"
        if not os.path.exists(p) or os.path.getsize(p) != 124059915:
            say("weights missing or wrong size:", p); sys.exit(1)
        shutil.copy(p, f"{DS_DIR}/{dst}")
        say("fetched", slug, src, "->", dst)

    say("dataset version:", cli(OWNER, "datasets", "version", "-p", DS_DIR, "-m", "add folds 2 and 3").splitlines()[-1][:200])
    for _ in range(40):
        if "ready" in cli(OWNER, "datasets", "status", DS):
            break
        time.sleep(30)
    else:
        say("dataset not ready"); sys.exit(1)
    say("dataset ready")

    new = [dst for _, dst in trained.values()]
    all_models = ["a0_f0.pth", "a1_f0.pth", "a2_f0.pth", "c1_f1.pth"] + new
    fused_models = ["a2_f0.pth", "c1_f1.pth"] + new
    w = lambda names: ",".join(f"knee-bp-weights/{x}" for x in names)
    sets = {"knee-bp-sub-f6": (w(all_models), f"box-pool cnxt-t 256, folds 0-3, label sets 1 / 7 / fused, mean of {len(all_models)} models"),
            "knee-bp-sub-f4": (w(fused_models), f"box-pool cnxt-t 256, folds 0-3, fused labels 1+7, mean of {len(fused_models)} models")}
    say("primary (all models):", all_models, "| ablation (fused only):", fused_models)
    for slug, (cls, _) in sets.items():
        r = subprocess.run([sys.executable, f"{HERE}/build_infer_kernel.py", "--slug", slug, "--user", "nishkarshsinghal",
                            "--weights-dataset", DS, "--cls", cls], capture_output=True, text=True)
        say("built", slug, r.stdout.strip()[:100], r.stderr.strip()[-200:])
        if r.returncode:
            sys.exit(1)
    research = [("knee-bp-sub-a1-sv", "box-pool cnxt-t 256, fold0, label set 7, single model, per-finding sharpening (medial/lateral OA 1.5, contusion/fracture 1.25)"),
                ("knee-bp-sub-a012", "box-pool cnxt-t 256, fold0, label sets 1 / 7 / fused, mean of 3 models"),
                ("knee-bp-sub-a1-s2", "box-pool cnxt-t 256, fold0, label set 7, single model, attention sharpening 2.0"),
                ("knee-bp-sub-a012c1", "box-pool cnxt-t 256, folds 0+1, label sets 1 / 7 / fused, mean of 4 models")]   # after the finals
    json.dump([{"slug": s, "msg": m} for s, (_, m) in sets.items()] + [{"slug": s, "msg": m} for s, m in research],
              open("D:/rsna_work/flow_owner_final_items.json", "w"), indent=1)
    r = subprocess.run([sys.executable, "-u", f"{HERE}/kaggle_flow.py", "--items", "D:/rsna_work/flow_owner_final_items.json",
                        "--token", OWNER, "--user", "nishkarshsinghal", "--state", "D:/rsna_work/flow_owner_final_state.json",
                        "--slots", "2", "--poll", "45", "--max-hours", "20"])
    say("final flow exit code", r.returncode)


if __name__ == "__main__":
    main()
