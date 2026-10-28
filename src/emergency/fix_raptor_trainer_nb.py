import json
import gzip
import base64
import os

def main():
    py_path = r'd:\Coding\kaggle\RSNA\notebooks\trainer_raptor_ft\train_knee.py'
    py_content = open(py_path, 'rb').read()
    b64_gz = base64.b64encode(gzip.compress(py_content)).decode('ascii')
    print('Compressed length:', len(b64_gz))

    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\trainer_raptor_ft\rsna-knee-raptor-ft-trainer.ipynb'
    nb = json.load(open(nb_path, encoding='utf-8'))

    code = f'''import glob, os, shutil, subprocess, sys, torch, base64, gzip
import pandas as pd

print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Device name:', torch.cuda.get_device_name(0))
    subprocess.run(['nvidia-smi'], check=False)

def find_file(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f'Could not find file matching: {{pattern}}'
    return hits[0]

# 1. Unpack train_knee.py directly
B64_GZ = \"\"\"{b64_gz}\"\"\"
with open('train_knee.py', 'wb') as f:
    f.write(gzip.decompress(base64.b64decode(B64_GZ)))
assert os.path.exists('train_knee.py') and os.path.getsize('train_knee.py') > 50000, 'train_knee.py unpack failed'
print('Successfully unpacked train_knee.py:', os.path.getsize('train_knee.py'), 'bytes', flush=True)

# 2. Convert CSV targets to Parquet
csv_path = find_file('/kaggle/input/**/training_labels_v6_vision_pure.csv')
print(f'Found targets at: {{csv_path}}', flush=True)
df = pd.read_csv(csv_path)
pq_path = '/kaggle/working/labels.parquet'
df.to_parquet(pq_path, index=False)
print(f'Prepared {{pq_path}} with {{len(df)}} studies.', flush=True)

# 3. Locate init checkpoint
init_pt = find_file('/kaggle/input/**/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')
print(f'Using init checkpoint: {{init_pt}}', flush=True)

# 4. Run fine-tuning with fp16 on T4
cmd = [
    sys.executable, 'train_knee.py',
    '--arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',
    '--res', '384',
    '--init_from', init_pt,
    '--ckpt', 'none',
    '--labels', pq_path,
    '--epochs', '2',
    '--bs', '4',
    '--k', '12',
    '--k_eval', '24',
    '--grad_ckpt',
    '--fp16',
    '--bb_lr', '1.5e-5',
    '--head_lr', '3e-4',
    '--wd', '0.05',
    '--tag', 'raptor_vision_pure',
    '--topk', '2',
    '--workers', '4'
]
print('Running:', ' '.join(cmd), flush=True)
rc = subprocess.run(cmd).returncode
print('Trainer exit code:', rc, flush=True)
assert rc == 0, f'Trainer failed with exit code {{rc}}'

# 5. List outputs
print('Outputs in /kaggle/working/:')
for f in os.listdir('/kaggle/working'):
    print(' ', f, os.path.getsize(os.path.join('/kaggle/working', f)), 'bytes')
'''

    nb['cells'][1]['source'] = [code]
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print('Notebook updated successfully!')

if __name__ == '__main__':
    main()
