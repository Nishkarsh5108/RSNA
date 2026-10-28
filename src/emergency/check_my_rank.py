import csv
import glob

csvs = glob.glob(r'D:\rsna_work\tmp_lb\*.csv')
if not csvs:
    print('No csv found in tmp_lb')
    exit(0)

with open(csvs[0], encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

print(f'Total teams on LB: {len(reader)}')
my_rank = None
my_score = None
for idx, r in enumerate(reader, 1):
    name = r.get('TeamName', '')
    if 'hent' in name.lower() or 'nishkarsh' in name.lower():
        my_rank = idx
        my_score = r.get('Score')
        print(f'Our Team: Rank #{idx} | Team: {name} | Score: {my_score}')

if my_rank:
    # Print teams around us
    start = max(0, my_rank - 6)
    end = min(len(reader), my_rank + 5)
    print('\n--- Teams Around Us ---')
    for i in range(start, end):
        r = reader[i]
        print(f'Rank #{i+1:3d} | Score: {r.get("Score")} | {r.get("TeamName")}')
