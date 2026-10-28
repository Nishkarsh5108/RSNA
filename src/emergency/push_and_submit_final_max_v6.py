import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
MAX_KERNEL = "nishkarshsinghal/rsna-knee-final-max"
STRICT_KERNEL = "nishkarshsinghal/rsna-knee-final-strict-v2"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = "final-max v5: dual coatnet + r6 + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"

def run_cmd(args):
    res = subprocess.run([PYTHON, "kg.py"] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for GPU slot: checking {STRICT_KERNEL}...", flush=True)

# Step 1: Wait for strict commit run to finish and free GPU slot
for i in range(50):
    ret, out = run_cmd(["kernels", "status", STRICT_KERNEL])
    ts = datetime.now().strftime('%H:%M:%S')
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    print(f"[{ts}] Strict check {i+1}/50: {last_line}", flush=True)
    if "KernelWorkerStatus.COMPLETE" in out or "KernelWorkerStatus.ERROR" in out:
        print(f"[{ts}] Strict commit run concluded ({last_line}). GPU slot freed!", flush=True)
        break
    time.sleep(25)

time.sleep(15)  # brief cool-off for Kaggle session teardown

# Step 2: Push final-max (Version 6)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Pushing final-max (Version 6)...", flush=True)
p_ret, p_out = run_cmd(["kernels", "push", "-p", r"notebooks\candidate_A\final_max"])
print(p_out, flush=True)

# Step 3: Monitor final-max commit run
print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring final-max commit run...", flush=True)
for i in range(100):
    ret, out = run_cmd(["kernels", "status", MAX_KERNEL])
    ts = datetime.now().strftime('%H:%M:%S')
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    print(f"[{ts}] final-max check {i+1}/100: {last_line}", flush=True)
    
    if "KernelWorkerStatus.COMPLETE" in out:
        print(f"\n[{ts}] >>> FINAL-MAX COMMIT RUN COMPLETED SUCCESSFULLY! <<<", flush=True)
        print(f"Submitting version 6 to {COMPETITION}...", flush=True)
        sub_args = [
            "competitions", "submit", COMPETITION,
            "-k", MAX_KERNEL,
            "-v", "6",
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
        print(f"\n[{ts}] >>> ERROR: final-max commit run encountered an error! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
        
    time.sleep(25)

sys.exit(1)
