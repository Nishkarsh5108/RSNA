import subprocess
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\nishk\anaconda3\envs\torch_gpu\python.exe"
MAX_KERNEL = "nishkarshsinghal/rsna-knee-final-max"
COMPETITION = "rsna-knee-abnormality-detection"
MSG_MAX = "final-max v5: dual coatnet + r6 + s6ft_specialist (0.9402) + s6ft_v1 (0.9864 MCL)"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_kg(args):
    cmd = [PYTHON, "kg.py"] + args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return res.returncode, res.stdout + res.stderr

log("=== STARTING FINAL-MAX WATCHER & SUBMITTER ===")

# Wait up to 300 checks * 25s (~2 hours) for final-max to transition QUEUED -> RUNNING -> COMPLETE
for i in range(1, 301):
    ret, out = run_kg(["kernels", "status", MAX_KERNEL])
    last_line = out.strip().splitlines()[-1] if out.strip() else 'No output'
    log(f"{MAX_KERNEL} status [{i}/300]: {last_line}")
    
    if "KernelWorkerStatus.COMPLETE" in out:
        log("\n>>> FINAL-MAX COMMIT RUN COMPLETED SUCCESSFULLY! <<<")
        log("Submitting final-max v9 to competition...")
        s_ret, s_out = run_kg(["competitions", "submit", COMPETITION, "-k", MAX_KERNEL, "-v", "9", "-f", "submission.csv", "-m", MSG_MAX])
        log(f"Submission Output:\n{s_out}")
        
        # Verify competition submissions list
        log("\nChecking competition submissions list:")
        _, sub_list = run_kg(["competitions", "submissions", COMPETITION])
        log(sub_list)
        sys.exit(0)
        
    elif "KernelWorkerStatus.ERROR" in out:
        log(f"\n>>> ERROR: final-max encountered an error! Downloading logs... <<<")
        run_kg(["kernels", "output", MAX_KERNEL])
        sys.exit(1)
        
    # Every 10 checks (~4.5 minutes), log current competition submissions to watch for scoring
    if i % 10 == 0:
        _, sub_out = run_kg(["competitions", "submissions", COMPETITION])
        first_few = '\n'.join(sub_out.strip().splitlines()[:6])
        log(f"Competitions check:\n{first_few}")
        
    time.sleep(25)

log("Timed out after 300 checks.")
sys.exit(1)
