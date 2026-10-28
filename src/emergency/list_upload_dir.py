import os
from pathlib import Path

p = Path("D:/rsna_work/kaggle_upload")
if p.exists():
    for f in p.iterdir():
        print(f.name, "is_dir:", f.is_dir())
else:
    print("Does not exist")
