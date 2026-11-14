"""Push inference notebooks on one Kaggle account, wait for each to finish, and late-submit each exactly once.

One process, no sub-shells (so stopping it stops everything), and a state file so a restart never submits twice.
The token file is read here and handed to the CLI through the child's environment only; nothing is printed.

  python kaggle_flow.py --items items.json --token "C:/Users/<me>/.kaggle/<file>.txt" --user <account> --state state.json

items.json: [{"slug": "knee-bp-sub-a2", "msg": "submission description"}, ...]   (notebook folders in D:/rsna_work/kaggle_runs)
"""
import argparse
import json
import os
import re
import subprocess
import time

K = "D:/rsna_work/kaggle_cli_venv/Scripts/kaggle.exe"
COMP = "rsna-knee-abnormality-detection"
RUNS = "D:/rsna_work/kaggle_runs"


def keep_awake():
    """Windows: no idle sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED; ends with the process).
    Closing the lid or a manual sleep still sleeps the machine, and the polling then resumes on wake."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


def cli(token_file, *args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KAGGLE_API_TOKEN=open(token_file).read().strip())
    p = subprocess.run([K, *args], capture_output=True, text=True, env=env)
    return "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "outdated" not in l and l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--slots", type=int, default=2, help="GPU sessions this account may have at once")
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--max-hours", type=float, default=24)
    a = ap.parse_args()
    items = json.load(open(a.items, encoding="utf-8"))
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    log_path = a.state.replace(".json", ".log")

    def say(*m):
        line = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in m)
        print(line, flush=True)
        open(log_path, "a", encoding="utf-8").write(line + "\n")

    def save():
        json.dump(state, open(a.state, "w"), indent=1)

    t0 = time.time()
    keep_awake()
    say(f"start: {len(items)} items, account {a.user}, state {state}")
    while time.time() - t0 < a.max_hours * 3600:
        active = [i for i in items if state.get(i["slug"], {}).get("version") and not state[i["slug"]].get("done")]
        # 1. submit what has finished
        for it in active:
            s = state[it["slug"]]
            out = cli(a.token, "kernels", "status", f"{a.user}/{it['slug']}/{s['version']}")
            status = re.search(r'KernelWorkerStatus\.(\w+)', out)
            status = status.group(1) if status else "?"
            if status == "COMPLETE":
                res = cli(a.token, "competitions", "submit", COMP, "-k", f"{a.user}/{it['slug']}", "-v", str(s["version"]),
                          "-f", "submission.csv", "-m", it["msg"])
                say("submit", it["slug"], "v", s["version"], "->", res.splitlines()[-1][:160] if res else "(no output)")
                if "submission" in res.lower() and "error" not in res.lower():
                    s["done"] = "submitted"
                else:
                    s["submit_failures"] = s.get("submit_failures", 0) + 1
                    if s["submit_failures"] >= 3:
                        s["done"] = "submit_failed"
                save()
            elif status in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                say("notebook", it["slug"], "ended with", status, "- not submitted")
                s["done"] = status
                save()
        # 2. push the next notebook when a slot is free
        active = [i for i in items if state.get(i["slug"], {}).get("version") and not state[i["slug"]].get("done")]
        pending = [i for i in items if not state.get(i["slug"], {}).get("version") and state.get(i["slug"], {}).get("done") is None]
        if pending and len(active) < a.slots:
            it = pending[0]
            out = cli(a.token, "kernels", "push", "-p", f"{RUNS}/{it['slug']}", "--accelerator", "NvidiaTeslaT4")
            m = re.search(r"Kernel version (\d+) successfully pushed", out)
            if m:
                state[it["slug"]] = {"version": int(m.group(1)), "pushed_at": time.strftime("%m-%d %H:%M:%S")}
                save()
                say("pushed", it["slug"], "version", m.group(1))
            elif "Maximum batch GPU session count" in out:
                pass                                           # both slots busy: try again next round
            else:
                s = state.setdefault(it["slug"], {})
                s["push_failures"] = s.get("push_failures", 0) + 1
                say("push", it["slug"], "failed:", out[-200:].replace("\n", " "))
                if s["push_failures"] >= 4:
                    s["done"] = "push_failed"
                save()
        if all(state.get(i["slug"], {}).get("done") for i in items):
            say("all items finished:", {k: v.get("done") for k, v in state.items()})
            return
        time.sleep(a.poll)
    say("time limit reached; state", state)


if __name__ == "__main__":
    main()
