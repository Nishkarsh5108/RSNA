import subprocess
import sys

cmd = [
    r"D:\rsna_work\kaggle_cli_venv\Scripts\python.exe",
    "kg.py",
    "competitions",
    "submit",
    "rsna-knee-abnormality-detection",
    "-k",
    "nishkarshsinghal/rsna-knee-final-strict-v2",
    "-v",
    "2",
    "-f",
    "submission.csv",
    "-m",
    "final-strict-v2: raptor 384 + s6 + s6ft_v2 + s6ft_v1 + s6fall inpass t2"
]
print("Running submission command for final-strict-v2...")
res = subprocess.run(cmd)
print("Exit code:", res.returncode)
sys.exit(res.returncode)
