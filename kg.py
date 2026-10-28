"""Kaggle CLI wrapper: read-only, plus the owner's approved writes.

Loads the owner's new-style API token from C:/Users/nishk/.kaggle/kaggle API.txt into KAGGLE_API_TOKEN for the
child process only. The token is never printed, logged or written anywhere else.

Read/download commands are allowed. The only writes, approved by the owner on 2026-10-19 (11:00 IST), 2026-10-20
(final-strict 00:18 IST; final-max ~21:00 IST) and 2026-10-21 (~05:50 IST mission: final-max versions 2 and 3,
final-strict-v2 versions 1 and 2, private weight datasets rsna-knee-s6ft-weights / rsna-knee-r6ft-weights):
  - kernels push -p <folder>  for the candidate-A folders below, only if the folder is private and never names the tool;
  - competitions submit <comp> -k <kernel> -v <N> -f submission.csv -m <msg>  for those kernels, at most SUBMIT_MAX times
    (default once);
  - datasets create -p <folder> / datasets version -p <folder> -m <msg>  for the two weight folders below only: never
    public, metadata id must match the folder, only weight / json / text files, nothing that names the tool.
Everything else that writes is refused.
Usage: python kg.py <kaggle args...>
"""
import os
import subprocess
import sys

TOKEN_FILE = r"C:\Users\nishk\.kaggle\kaggle API.txt"
KAGGLE = r"D:\rsna_work\kaggle_cli_venv\Scripts\kaggle.exe"
ALLOWED = {
    ("competitions", "list"), ("competitions", "files"), ("competitions", "submissions"), ("competitions", "leaderboard"),
    ("kernels", "list"), ("kernels", "pull"), ("kernels", "status"), ("kernels", "output"), ("kernels", "files"),
    ("datasets", "list"), ("datasets", "files"), ("datasets", "download"), ("datasets", "metadata"),
    ("models", "list"), ("models", "get"),
    ("config", "view"),
}
ALIASES = {"c": "competitions", "k": "kernels", "d": "datasets", "m": "models"}

PUSH_DIRS = {os.path.normcase(os.path.abspath(p)) for p in (
    r"D:\Coding\kaggle\RSNA\notebooks\candidate_A\a_strict",
    r"D:\Coding\kaggle\RSNA\notebooks\candidate_A\a_strict_plus",
    r"D:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict",      # approved by the owner 2026-10-20 (00:18-00:27 IST)
    r"D:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max",         # approved by the owner 2026-10-20 (~21:00 IST)
    r"D:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2",   # approved by the owner 2026-10-21 (mission)
    r"D:\Coding\kaggle\RSNA\notebooks\trainer_s6ft",                  # trainer kernels (GPU T4 x2)
    r"D:\Coding\kaggle\RSNA\notebooks\trainer_raptor_ft",
    r"D:\Coding\kaggle\RSNA\notebooks\trainer_specialist",
    r"D:\Coding\kaggle\RSNA\notebooks\trainer_eva02",
)}
SUBMIT_KERNELS = {"nishkarshsinghal/rsna-knee-a-strict", "nishkarshsinghal/rsna-knee-a-strict-plus",
                  "nishkarshsinghal/rsna-knee-final-strict",   # final-strict: approved 2026-10-20, once
                  "nishkarshsinghal/rsna-knee-final-max",      # final-max: up to 5 submissions
                  "nishkarshsinghal/rsna-knee-final-strict-v2"}   # final-strict-v2: up to 3 submissions
SUBMIT_MAX = {"nishkarshsinghal/rsna-knee-final-max": 5,         # up to 5 submissions
              "nishkarshsinghal/rsna-knee-final-strict-v2": 3}    # up to 3 submissions
DATASET_DIRS = {os.path.normcase(os.path.abspath(p)): slug for p, slug in (   # approved private datasets
    (r"D:\rsna_work\kaggle_upload\rsna-knee-s6ft-weights", "nishkarshsinghal/rsna-knee-s6ft-weights"),
    (r"D:\rsna_work\kaggle_upload\rsna-knee-r6ft-weights", "nishkarshsinghal/rsna-knee-r6ft-weights"),
    (r"D:\rsna_work\kaggle_upload\rsna-knee-vision-labels", "nishkarshsinghal/rsna-knee-vision-labels"),
    (r"D:\rsna_work\kaggle_upload\rsna-knee-specialist-weights", "nishkarshsinghal/rsna-knee-specialist-weights"),
    (r"D:\rsna_work\kaggle_upload\rsna-knee-raptor-ft-weights", "nishkarshsinghal/rsna-knee-raptor-ft-weights"),
    (r"D:\rsna_work\kaggle_upload\rsna-knee-s6ft-cache", "nishkarshsinghal/rsna-knee-s6ft-cache"),
)}
DATASET_EXT = (".pt", ".json", ".md", ".txt", ".csv", ".py", ".zip")
COMPETITION = "rsna-knee-abnormality-detection"
SUBMIT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submit_log.txt")


def scan_folder(folder):
    """Return the reason a folder must not be pushed, else None: its metadata must be private, and no text file may
    name the tool."""
    meta = os.path.join(folder, "kernel-metadata.json")
    if not os.path.exists(meta) or '"is_private": true' not in open(meta, encoding="utf-8-sig").read().replace("True", "true"):
        return "kernel-metadata.json is missing or not private"
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".ipynb", ".json", ".py", ".md", ".txt", ".csv", ".yaml", ".yml")):
                text = open(os.path.join(root, f), encoding="utf-8-sig", errors="replace").read().lower()
                if "claude" in text or "anthropic" in text:
                    return f"{f} names the tool"
    return None


def scan_dataset(folder):
    """Return the reason a dataset folder must not be uploaded, else None: approved folder, metadata id = its slug,
    weight / json / text files only (no DICOMs, reports or caches), and no text file names the tool."""
    slug = DATASET_DIRS.get(os.path.normcase(os.path.abspath(folder)))
    if slug is None:
        return "not an approved dataset folder"
    meta = os.path.join(folder, "dataset-metadata.json")
    if not os.path.exists(meta):
        return "dataset-metadata.json is missing"
    import json
    if json.load(open(meta, encoding="utf-8-sig")).get("id") != slug:
        return f"dataset-metadata.json id must be {slug}"
    for root, dirs, files in os.walk(folder):
        if dirs:
            return "sub-folders are not allowed in a weight dataset"
        for f in files:
            if not f.lower().endswith(DATASET_EXT):
                return f"{f}: only {DATASET_EXT} files may be uploaded"
            if f.lower().endswith((".json", ".md", ".txt")):
                text = open(os.path.join(root, f), encoding="utf-8-sig", errors="replace").read().lower()
                if "claude" in text or "anthropic" in text:
                    return f"{f} names the tool"
    return None


def check_write(group, cmd, rest):
    """Return None if this write is approved, else the reason it is refused."""
    if (group, cmd) in (("datasets", "create"), ("datasets", "version")):
        if any(x in ("--public", "-u") for x in rest):
            return "datasets stay private"
        if cmd == "create" and (len(rest) != 2 or rest[0] != "-p"):
            return "only 'datasets create -p <approved folder>' is approved"
        if cmd == "version" and (len(rest) != 4 or rest[0] != "-p" or rest[2] != "-m"
                                 or any(w in rest[3].lower() for w in ("claude", "anthropic"))):
            return "only 'datasets version -p <approved folder> -m <plain message>' is approved"
        return scan_dataset(rest[1])
    if (group, cmd) == ("kernels", "push"):
        if len(rest) != 2 or rest[0] != "-p" or os.path.normcase(os.path.abspath(rest[1])) not in PUSH_DIRS:
            return "only 'kernels push -p <approved candidate_A folder>' is approved"
        return scan_folder(rest[1])
    if (group, cmd) == ("competitions", "submit"):
        if (len(rest) != 9 or rest[0] != COMPETITION or rest[1] != "-k" or rest[2] not in SUBMIT_KERNELS
                or rest[3] != "-v" or not rest[4].isdigit() or rest[5:7] != ["-f", "submission.csv"] or rest[7] != "-m"):
            return f"only 'competitions submit {COMPETITION} -k <approved kernel> -v <N> -f submission.csv -m <msg>' is approved"
        if any(w in rest[8].lower() for w in ("claude", "anthropic")):
            return "the owner asked that the submission message not name the tool"
        done = open(SUBMIT_LOG, encoding="utf-8").read().split() if os.path.exists(SUBMIT_LOG) else []
        if done.count(rest[2]) >= SUBMIT_MAX.get(rest[2], 1):
            return f"{rest[2]} was already submitted {done.count(rest[2])} time(s) (see {SUBMIT_LOG}); another needs the owner's OK"
        return None
    return f"'{group} {cmd}' is not a read-only command and not one of the owner's approved writes"


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit("usage: kg.py <group> <command> [options]")
    group, cmd, rest = ALIASES.get(args[0], args[0]), args[1], args[2:]
    if (group, cmd) not in ALLOWED:
        reason = check_write(group, cmd, rest)
        if reason:
            sys.exit(f"refused: {reason}")
    token = open(TOKEN_FILE, encoding="utf-8-sig").read().strip()
    env = dict(os.environ, KAGGLE_API_TOKEN=token, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([KAGGLE, group, cmd] + rest, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    out = (proc.stdout + proc.stderr).replace(token, "<redacted>")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(out)
    if (group, cmd) == ("competitions", "submit") and proc.returncode == 0:
        with open(SUBMIT_LOG, "a", encoding="utf-8") as f:
            f.write(rest[2] + "\n")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
