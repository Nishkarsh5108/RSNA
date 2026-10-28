src = open("C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6/compact_infer.py", encoding="utf-8").read()
lines = src.splitlines()
for idx, l in enumerate(lines):
    if "flip" in l.lower() and ("def " in l or "forward" in l or "score" in l or "tta" in l or "handle" in l):
        print(f"{idx}: {l}")
        for j in range(max(0, idx - 2), min(len(lines), idx + 15)):
            print(f"  {lines[j]}")
