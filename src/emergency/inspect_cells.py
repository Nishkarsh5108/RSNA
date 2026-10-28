import json

def main():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))
    
    print('--- Cell 4 (execution of raptor) ---')
    print(''.join(nb['cells'][4]['source']))

    print('--- Cell 12 (blend) ---')
    print(''.join(nb['cells'][12]['source'])[:2000])

    print('--- Cell 18 (final execution / blend) ---')
    print(''.join(nb['cells'][18]['source'])[:2000])

if __name__ == '__main__':
    main()
