import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
MAX_KERNEL = "nishkarshsinghal/rsna-knee-final-max"
STRICT_KERNEL = "nishkarshsinghal/rsna-knee-final-strict-v2"
COMPETITION = "rsna-knee-abnormality-detection"
MSG = "final-strict-v3: dual coatnet + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"

def run_cmd(args):
    res = subprocess.run([PYTHON, "kg.py"] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for final-max commit run to complete to free GPU slot...", flush=True)

# Step 1: Wait for final-max v6 commit run to finish
for i in range(50):
    ret, out = run_cmd(["kernels", "status", MAX_KERNEL])
    ts = datetime.now().strftime('%H:%M:%S')
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    print(f"[{ts}] final-max status check {i+1}/50: {last_line}", flush=True)
    if "KernelWorkerStatus.COMPLETE" in out or "KernelWorkerStatus.ERROR" in out:
        print(f"[{ts}] final-max commit run concluded ({last_line}). GPU slot freed!", flush=True)
        break
    time.sleep(25)

time.sleep(20)  # brief cool-off for Kaggle GPU slot release

# Step 2: Push final-strict-v2 (Version 5)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Pushing final-strict-v2 (Version 5)...", flush=True)
p_ret, p_out = run_cmd(["kernels", "push", "-p", r"notebooks\candidate_A\final_strict_v2"])
print(p_out, flush=True)

# Step 3: Monitor final-strict commit run and submit
print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring final-strict commit run...", flush=True)
for i in range(100):
    ret, out = run_cmd(["kernels", "status", STRICT_KERNEL])
    ts = datetime.now().strftime('%H:%M:%S')
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    print(f"[{ts}] final-strict check {i+1}/100: {last_line}", flush=True)
    
    if "KernelWorkerStatus.COMPLETE" in out:
        print(f"\n[{ts}] >>> FINAL-STRICT COMMIT RUN COMPLETED SUCCESSFULLY! <<<", flush=True)
        print(f"Submitting version 5 to {COMPETITION}...", flush=True)
        sub_args = [
            "competitions", "submit", COMPETITION,
            "-k", STRICT_KERNEL,
            "-v", "5",
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
        print(f"\n[{ts}] >>> ERROR: final-strict commit run encountered an error! <<<", flush=True)
        print(out, flush=True)
        sys.exit(1)
        
    time.sleep(25)

sys.exit(1)
