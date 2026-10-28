import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
KERNEL = "nishkarshsinghal/rsna-knee-final-max"
COMPETITION = "rsna-knee-abnormality-detection"
VERSION = "5"
MSG = "final-max v5: dual coatnet + r6 + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"

def run_cmd(args):
    res = subprocess.run([PYTHON, "kg.py"] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting auto-submit watcher for {KERNEL} (Version {VERSION})...", flush=True)

max_attempts = 100  # ~45 minutes
for i in range(max_attempts):
    ret, out = run_cmd(["kernels", "status", KERNEL])
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] Check {i+1}/{max_attempts}: {last_line}", flush=True)
    
    if "KernelWorkerStatus.COMPLETE" in out:
        print(f"\n[{ts}] >>> KERNEL COMMIT RUN COMPLETED SUCCESSFULLY! <<<", flush=True)
        print(f"Submitting version {VERSION} to {COMPETITION}...", flush=True)
        sub_args = [
            "competitions", "submit", COMPETITION,
            "-k", KERNEL,
            "-v", VERSION,
            "-f", "submission.csv",
            "-m", MSG
        ]
        s_ret, s_out = run_cmd(sub_args)
        print("Submission Output:", flush=True)
        print(s_out, flush=True)
        
        # Verify submission list
        print("\nChecking competition submissions list:", flush=True)
        _, sub_list = run_cmd(["competitions", "submissions", COMPETITION])
        print(sub_list, flush=True)
        sys.exit(0)
    elif "KernelWorkerStatus.ERROR" in out:
        print(f"\n[{ts}] >>> ERROR: Kernel commit run encountered an error! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
    elif "KernelWorkerStatus.CANCELLED" in out:
        print(f"\n[{ts}] >>> ERROR: Kernel commit run was cancelled! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
        
    time.sleep(25)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Timed out waiting for kernel completion.", flush=True)
sys.exit(1)
