import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
KERNEL = "nishkarshsinghal/rsna-knee-final-max"
COMPETITION = "rsna-knee-abnormality-detection"
VERSION = "10"
MSG = "final-max v10: master blend dual coatnet + r6 + s6fall + calibrated specialists"

def run_cmd(args):
    res = subprocess.run([PYTHON, "kg.py"] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

print(f"[{datetime.now().strftime('%H:%M:%S')}] Pushing final-max Version {VERSION} (Master Blend)...", flush=True)

# Step 1: Push kernel
p_ret, p_out = run_cmd(["kernels", "push", "-p", r"notebooks\candidate_A\final_max"])
print(p_out, flush=True)
if p_ret != 0:
    print(f"Error pushing kernel: exit code {p_ret}")
    sys.exit(1)

# Step 2: Monitor commit run
print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring commit run on Kaggle GPU...", flush=True)
for i in range(100):  # up to 45 min
    ret, out = run_cmd(["kernels", "status", KERNEL])
    ts = datetime.now().strftime('%H:%M:%S')
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    print(f"[{ts}] Check {i+1}/100: {last_line}", flush=True)
    
    if "KernelWorkerStatus.COMPLETE" in out:
        print(f"\n[{ts}] >>> COMMIT RUN FINISHED SUCCESSFULLY! <<<", flush=True)
        print(f"Submitting Version {VERSION} to {COMPETITION}...", flush=True)
        sub_args = [
            "competitions", "submit", COMPETITION,
            "-k", KERNEL,
            "-v", VERSION,
            "-f", "submission.csv",
            "-m", MSG
        ]
        s_ret, s_out = run_cmd(sub_args)
        print("Submission output:", flush=True)
        print(s_out, flush=True)
        
        # Verify competition submissions list
        print("\nVerifying competition submissions:", flush=True)
        _, sub_list = run_cmd(["competitions", "submissions", COMPETITION])
        print(sub_list, flush=True)
        sys.exit(0)
    elif "KernelWorkerStatus.ERROR" in out:
        print(f"\n[{ts}] >>> ERROR in commit run! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
    elif "KernelWorkerStatus.CANCELLED" in out:
        print(f"\n[{ts}] >>> Commit run was cancelled! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
        
    time.sleep(25)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Timed out waiting for commit run.", flush=True)
sys.exit(1)
