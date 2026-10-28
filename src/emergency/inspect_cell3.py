import json

def main():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))
    cell3 = ''.join(nb['cells'][3]['source'])
    for line in cell3.splitlines():
        if '.pt' in line or 'swa' in line or 'init' in line or 'CKPT' in line:
            print(line)

if __name__ == '__main__':
    main()
