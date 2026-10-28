import zipfile
import csv
import io
import sys

sys.stdout.reconfigure(encoding='utf-8')

z = zipfile.ZipFile("D:/Coding/kaggle/RSNA/rsna-knee-abnormality-detection.zip")
fname = z.namelist()[0]
print("File inside zip:", fname)

with z.open(fname) as f:
    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
    count = 0
    matches = []
    top20 = []
    headers = None
    for row in reader:
        count += 1
        if count == 1:
            headers = list(row.keys())
        if count <= 20:
            top20.append(row)
        name = row.get("TeamName", "")
        if any(x in name.lower() for x in ["hent", "nishkarsh", "singhal"]):
            matches.append((count, row))

print("Total teams:", count)
print("Headers:", headers)
print("\nTop 5 teams:")
for r in top20[:5]:
    print(r)

print("\nOur team matches:")
for rank_idx, r in matches:
    print(f"Row/Rank {rank_idx}: {r}")
