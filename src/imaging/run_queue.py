"""Run a queue of training experiments sequentially (one GPU job at a time).

The queue is RE-READ from output/imaging_v5/queue.json before every run, so runs can be
added, removed or re-ordered while the queue is running. Entry format:
  {"run_id": "E4_...", "args": ["--slots", "0,1,2,3,4,5"], "note": "why this run exists",
   "requires": ["path that must exist first"]}          (optional; otherwise the next ready run goes first)
A run is skipped if its summary.json exists; an interrupted run is resumed (--resume).
A run that fails (non-zero exit) is not retried automatically (listed in the log).
--wait-pid PID: first wait for another (training) process to exit (hand-over from an older queue).
--queue FILE: use another queue file, e.g. the deferred analysis runs:
    python src/imaging/run_queue.py --queue output/imaging_v5/queue_deferred.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

R = "D:/Coding/kaggle/RSNA"
PY = sys.executable
RUNS = f"{R}/output/imaging_v5/runs"
LOGS = f"{R}/output/imaging_v5/logs"
QFILE = f"{R}/output/imaging_v5/queue.json"
COMMON = ["--epochs", "8", "--workers", "6"]


def pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--queue", default=QFILE, help="queue file (e.g. output/imaging_v5/queue_deferred.json)")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.wait_pid:
        print(time.strftime("%H:%M:%S"), f"waiting for pid {a.wait_pid} to exit", flush=True)
        while pid_alive(a.wait_pid):
            time.sleep(30)
    failed, waiting_note = set(), None
    while True:
        queue = json.load(open(a.queue))
        todo = [q for q in queue if q["run_id"] not in failed and not os.path.exists(f"{RUNS}/{q['run_id']}/summary.json")]
        if not todo:
            print(time.strftime("%H:%M:%S"), "queue empty; failed:", sorted(failed), flush=True)
            return
        ready = [q for q in todo if all(os.path.exists(p) for p in q.get("requires", []))]
        if not ready:                     # e.g. a cache that is still being built
            note = [q["run_id"] for q in todo]
            if note != waiting_note:
                print(time.strftime("%H:%M:%S"), "waiting for prerequisites of", note, flush=True)
                waiting_note = note
            time.sleep(60)
            continue
        q = ready[0]
        run_id = q["run_id"]
        cmd = [PY, f"{R}/src/imaging/train.py", "--run-id", run_id, *q.get("common", COMMON), *q["args"]]
        if os.path.exists(f"{RUNS}/{run_id}/ckpt_last.pt"):
            cmd.append("--resume")
        print(time.strftime("%H:%M:%S"), "start", run_id, " ".join(q["args"]), flush=True)
        with open(f"{LOGS}/{run_id}.log", "a") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M:%S"), "end", run_id, "rc", rc, flush=True)
        if rc != 0 or not os.path.exists(f"{RUNS}/{run_id}/summary.json"):
            failed.add(run_id)


if __name__ == "__main__":
    main()
