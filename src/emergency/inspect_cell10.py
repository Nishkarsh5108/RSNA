import json

def main():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))
    print('--- Cell 10 ---')
    print(''.join(nb['cells'][10]['source']))

if __name__ == '__main__':
    main()
