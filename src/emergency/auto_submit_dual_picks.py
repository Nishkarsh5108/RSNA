import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
MAX_KERNEL = "nishkarshsinghal/rsna-knee-final-max"
STRICT_KERNEL = "nishkarshsinghal/rsna-knee-final-strict-v2"
COMPETITION = "rsna-knee-abnormality-detection"

MSG_STRICT = "final-strict-v3: dual coatnet + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"
MSG_MAX = "final-max v5: dual coatnet + r6 + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_kg(args):
    cmd = [PYTHON, "kg.py"] + args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

def wait_for_kernel(kernel_slug, max_checks=120, sleep_sec=25):
    for i in range(1, max_checks + 1):
        ret, out = run_kg(["kernels", "status", kernel_slug])
        last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
        log(f"{kernel_slug} status [{i}/{max_checks}]: {last_line}")
        if "KernelWorkerStatus.COMPLETE" in out:
            return "COMPLETE"
        if "KernelWorkerStatus.ERROR" in out:
            return "ERROR"
        time.sleep(sleep_sec)
    return "TIMEOUT"

def get_latest_version(kernel_slug):
    # Retrieve current version by pushing or checking submissions
    # For Kaggle kernels, version increments each push
    pass

log("=== STARTING DUAL-PICK ORCHESTRATOR ===")

# STEP 1: Check final-strict-v2 status
log("Checking current status of final-strict-v2...")
status = wait_for_kernel(STRICT_KERNEL, max_checks=60, sleep_sec=20)

if status == "COMPLETE":
    log("final-strict-v2 commit run SUCCEEDED! Submitting...")
    # Attempt submit version 5 or latest
    for v in ["5", "6"]:
        ret, out = run_kg(["competitions", "submit", COMPETITION, "-k", STRICT_KERNEL, "-v", v, "-f", "submission.csv", "-m", MSG_STRICT])
        log(f"Submit strict v{v} output:\n{out}")
        if "Successfully submitted" in out or "Your submission was accepted" in out or "pending" in out.lower():
            break
elif status == "ERROR":
    log("final-strict-v2 v5 encountered an error (likely unpatched infer_probs). Repushing patched v6...")
    time.sleep(15)
    ret, out = run_kg(["kernels", "push", "-p", r"notebooks\candidate_A\final_strict_v2"])
    log(f"Push v6 output: {out}")
    time.sleep(20)
    status_v6 = wait_for_kernel(STRICT_KERNEL, max_checks=80, sleep_sec=25)
    if status_v6 == "COMPLETE":
        log("final-strict-v2 v6 SUCCEEDED! Submitting...")
        ret, out = run_kg(["competitions", "submit", COMPETITION, "-k", STRICT_KERNEL, "-v", "6", "-f", "submission.csv", "-m", MSG_STRICT])
        log(f"Submit strict v6 output:\n{out}")
    else:
        log(f"final-strict-v2 v6 failed with status: {status_v6}")
        # Download log for diagnosis
        run_kg(["kernels", "output", STRICT_KERNEL])

# Cool down for GPU slot release
log("Cooling down 25s for GPU slot release before pushing final-max...")
time.sleep(25)

# STEP 2: Push and submit final-max (Version 7)
log("Pushing patched final-max (Version 7)...")
ret, out = run_kg(["kernels", "push", "-p", r"notebooks\candidate_A\final_max"])
log(f"Push final-max v7 output: {out}")
time.sleep(25)

status_max = wait_for_kernel(MAX_KERNEL, max_checks=80, sleep_sec=25)
if status_max == "COMPLETE":
    log("final-max v7 SUCCEEDED! Submitting...")
    ret, out = run_kg(["competitions", "submit", COMPETITION, "-k", MAX_KERNEL, "-v", "7", "-f", "submission.csv", "-m", MSG_MAX])
    log(f"Submit final-max v7 output:\n{out}")
else:
    log(f"final-max v7 failed with status: {status_max}")
    run_kg(["kernels", "output", MAX_KERNEL])

# STEP 3: Verify competition submissions table
log("Checking competition submissions table...")
_, out = run_kg(["competitions", "submissions", COMPETITION])
log(f"Current submissions:\n{out}")
log("=== DUAL-PICK ORCHESTRATOR COMPLETE ===")
