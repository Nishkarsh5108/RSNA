src = open("C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6/compact_infer.py", encoding="utf-8").read()
lines = src.splitlines()
start = None
for idx, l in enumerate(lines):
    if "def load_group(" in l:
        start = idx
        break
for i in range(start, start + 35):
    print(lines[i])
