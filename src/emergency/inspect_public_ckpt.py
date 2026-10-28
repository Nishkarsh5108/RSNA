"""Compare the layer layout of public fold checkpoints (R6 / R7 / S6) without running them.

torch.load(weights_only=True) uses the restricted unpickler: tensors, dicts, lists and primitives only, no code.
Used for plan 10c (fold 0 of R6, R7, S6).
Usage: python -I src/emergency/inspect_public_ckpt.py name=path [name=path ...]
"""
import sys
import zipfile

import torch


def flat(obj, prefix=""):
    """Yield (key, shape-or-value) for every leaf of nested dicts."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}{k}." if not isinstance(v, torch.Tensor) else f"{prefix}{k}")
    elif isinstance(obj, torch.Tensor):
        yield prefix, tuple(obj.shape)
    elif isinstance(obj, (list, tuple)) and len(obj) < 50:
        yield prefix.rstrip("."), repr(obj)[:200]
    else:
        yield prefix.rstrip("."), repr(obj)[:200]


ck = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    with zipfile.ZipFile(path) as z:
        print(f"== {name}: zip entries {len(z.namelist())}, first {z.namelist()[:3]}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    leaves = dict(flat(obj))
    ck[name] = (obj, leaves)
    top = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
    print(f"   top-level: {top}")
    tensors = {k: v for k, v in leaves.items() if isinstance(v, tuple)}
    print(f"   tensors {len(tensors)}, params {sum(int(torch.tensor(s).prod()) if s else 1 for s in tensors.values()):,}")
    non = {k: v for k, v in leaves.items() if not isinstance(v, tuple)}
    for k, v in list(non.items())[:40]:
        print(f"   meta {k} = {v}")
    heads = [k for k in tensors if not k.split(".")[-2:-1] or not any(s in k for s in ("stages.", "stem.", "downsample"))]
    for k in heads[:60]:
        print(f"   head-ish {k} {tensors[k]}")
    hits = [k for k in leaves if any(s in k.lower() for s in ("fm", "aux", "fast", "box", "det"))]
    print(f"   keys mentioning fm/aux/fast/box/det: {hits[:30]}")

names = list(ck)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = ck[names[i]][1], ck[names[j]][1]
        only_a = sorted(set(a) - set(b)); only_b = sorted(set(b) - set(a))
        diff_shape = sorted(k for k in set(a) & set(b) if isinstance(a[k], tuple) and a[k] != b[k])
        print(f"== {names[i]} vs {names[j]}: only {names[i]} {only_a[:20]}; only {names[j]} {only_b[:20]}; "
              f"shape differs {diff_shape[:20]}")
