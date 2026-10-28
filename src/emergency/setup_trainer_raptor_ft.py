import os
import json
import shutil

out_dir = r"D:\Coding\kaggle\RSNA\notebooks\trainer_raptor_ft"
os.makedirs(out_dir, exist_ok=True)

# Copy train_knee.py into the folder
src_train_knee = r"D:\rsna_work\public_weights\rsna-knee-s6fall-weights\training_code\coatnet\train_knee.py"
with open(src_train_knee, "r", encoding="utf-8") as f:
    train_knee_code = f.read()

with open(os.path.join(out_dir, "train_knee.py"), "w", encoding="utf-8") as f:
    f.write(train_knee_code)

# Metadata
meta = {
    "id": "nishkarshsinghal/rsna-knee-raptor-ft-trainer",
    "title": "rsna-knee-raptor-ft-trainer",
    "code_file": "rsna-knee-raptor-ft-trainer.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [
        "dreaddevelopment/knee-raptor-corpus",
        "dreaddevelopment/knee-raptor-corpus-ext",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "nishkarshsinghal/rsna-knee-vision-labels"
    ],
    "competition_sources": [
        "rsna-knee-abnormality-detection"
    ],
    "kernel_sources": [],
    "model_sources": []
}

with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# Notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# rsna knee raptor ft trainer\n",
                "\n",
                "fine-tuning coatnet-rmlp-2-rw-384 on pure vision-distilled targets (r6_oof + s6_oof + s6ft_v2) on kaggle gpu."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import glob, os, shutil, subprocess, sys, torch\n",
                "import pandas as pd\n",
                "\n",
                "print('PyTorch version:', torch.__version__)\n",
                "print('CUDA available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Device count:', torch.cuda.device_count())\n",
                "    print('Device name:', torch.cuda.get_device_name(0))\n",
                "    subprocess.run(['nvidia-smi'], check=False)\n",
                "\n",
                "def find_file(pattern):\n",
                "    hits = sorted(glob.glob(pattern, recursive=True))\n",
                "    assert hits, f'Could not find file matching: {pattern}'\n",
                "    return hits[0]\n",
                "\n",
                "# 1. Convert CSV targets to Parquet\n",
                "csv_path = find_file('/kaggle/input/**/training_labels_v6_vision_pure.csv')\n",
                "print(f'Found targets at: {csv_path}', flush=True)\n",
                "df = pd.read_csv(csv_path)\n",
                "pq_path = '/kaggle/working/labels.parquet'\n",
                "df.to_parquet(pq_path, index=False)\n",
                "print(f'Prepared {pq_path} with {len(df)} studies.', flush=True)\n",
                "\n",
                "# 2. Locate init checkpoint\n",
                "init_pt = find_file('/kaggle/input/**/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')\n",
                "print(f'Using init checkpoint: {init_pt}', flush=True)\n",
                "\n",
                "# 3. Locate train_knee.py\n",
                "script_path = 'train_knee.py'\n",
                "if not os.path.exists(script_path):\n",
                "    hits = glob.glob('/kaggle/input/**/train_knee.py', recursive=True)\n",
                "    if hits:\n",
                "        shutil.copy(hits[0], script_path)\n",
                "assert os.path.exists(script_path), 'Missing train_knee.py'\n",
                "\n",
                "# 4. Run fine-tuning\n",
                "cmd = [\n",
                "    sys.executable, script_path,\n",
                "    '--arch', 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',\n",
                "    '--res', '384',\n",
                "    '--init_from', init_pt,\n",
                "    '--labels', pq_path,\n",
                "    '--epochs', '2',\n",
                "    '--bs', '4',\n",
                "    '--k', '12',\n",
                "    '--k_eval', '24',\n",
                "    '--grad_ckpt',\n",
                "    '--bb_lr', '1.5e-5',\n",
                "    '--head_lr', '3e-4',\n",
                "    '--wd', '0.05',\n",
                "    '--tag', 'raptor_vision_pure',\n",
                "    '--topk', '2',\n",
                "    '--workers', '4'\n",
                "]\n",
                "print('Running:', ' '.join(cmd), flush=True)\n",
                "rc = subprocess.run(cmd).returncode\n",
                "print('Trainer exit code:', rc, flush=True)\n",
                "assert rc == 0, f'Trainer failed with exit code {rc}'\n",
                "\n",
                "# 5. List outputs\n",
                "print('Outputs in /kaggle/working/:')\n",
                "for f in os.listdir('/kaggle/working'):\n",
                "    print(' ', f, os.path.getsize(os.path.join('/kaggle/working', f)), 'bytes')\n"
            ]
        }
    ]
}

with open(os.path.join(out_dir, "rsna-knee-raptor-ft-trainer.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Successfully configured trainer_raptor_ft in", out_dir)
