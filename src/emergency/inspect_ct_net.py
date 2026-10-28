import inspect
import sys

sys.path.insert(0, "D:/rsna_work/public_code/s6_src")
import compact_train as CT

lines, start = inspect.getsourcelines(CT.Net)
for idx, l in enumerate(lines):
    if any(k in l for k in ["def _build_head", "def forward", "def pool"]):
        print(f"Line {idx}: {l[:70]}")
