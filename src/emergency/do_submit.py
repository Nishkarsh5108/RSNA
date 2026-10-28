import subprocess
import sys

cmd = [
    r"D:\rsna_work\kaggle_cli_venv\Scripts\python.exe",
    "kg.py",
    "competitions",
    "submit",
    "rsna-knee-abnormality-detection",
    "-k",
    "nishkarshsinghal/rsna-knee-final-max",
    "-v",
    "3",
    "-f",
    "submission.csv",
    "-m",
    "final-max v3: r6 + s6ft_v2 (0.9366) + s6ft_v1 + s6fall specialist inpass"
]
print("Running submission command...")
res = subprocess.run(cmd)
print("Exit code:", res.returncode)
sys.exit(res.returncode)
