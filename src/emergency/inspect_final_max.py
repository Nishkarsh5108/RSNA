import json

def main():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))
    print(f'Total cells: {len(nb["cells"])}')
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        first_line = src.split('\n')[0] if src else ''
        ctype = c.get('cell_type')
        nlines = len(src.splitlines())
        print(f'Cell {i} ({ctype}, {nlines} lines): {first_line[:70]}')

if __name__ == '__main__':
    main()
