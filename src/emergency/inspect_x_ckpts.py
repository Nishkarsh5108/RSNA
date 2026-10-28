import json

def main():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))
    cell9 = ''.join(nb['cells'][9]['source'])
    for i, line in enumerate(cell9.splitlines()):
        if '_x_ckpts' in line or 'S6FT_DIR' in line or 'S6F_DIR' in line:
            print(f'{i}: {line}')

if __name__ == '__main__':
    main()
