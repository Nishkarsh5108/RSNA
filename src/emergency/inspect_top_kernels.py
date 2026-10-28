import json
import glob
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

kernels = glob.glob(r'D:\rsna_work\tmp_kernels\*\*.ipynb')
for k in kernels:
    print('=' * 80)
    print('KERNEL:', os.path.basename(os.path.dirname(k)), '/', os.path.basename(k))
    try:
        nb = json.load(open(k, encoding='utf-8'))
    except Exception as e:
        print('Error reading:', e)
        continue
    
    # Print markdown description cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            text = ''.join(cell.get('source', []))
            if any(w in text.lower() for w in ['blend', 'model', 'weight', 'score', 'lb', 'ensemble']):
                print('--- Markdown Snippet ---')
                print(text[:400])
                print('...')
                break
    
    # Look for dataset sources, weights, and blending formula
    for idx, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if any(w in src for w in ['weights', 'blend', 'ARMS', 'WEIGHTS', 'pct_on', 'W_RAPTOR', 'W_R6']):
            print(f'--- Code Cell {idx} Snippet ---')
            for line in src.splitlines():
                if any(w in line for w in ['=', 'ARMS', 'weight', 'blend', 'model', 'W_']) and len(line) < 120:
                    print('  ', line.strip())
