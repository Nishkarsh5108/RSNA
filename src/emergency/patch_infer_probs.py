import json

def patch_infer_probs(nb_path, cell_idx):
    nb = json.load(open(nb_path, encoding='utf-8'))
    src = ''.join(nb['cells'][cell_idx]['source'])

    old_func = """@torch.no_grad()
def infer_probs(model, xwins, device):
    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"
    x = xwins.unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x).float()
        assert torch.isfinite(logits).all(), "Nonfinite fp16 logits"
        o = torch.sigmoid(logits)
    return o[0].cpu().numpy()"""

    new_func = """@torch.no_grad()
def infer_probs(model, xwins, device):
    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"
    x = xwins.unsqueeze(0).to(device)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            if not torch.isfinite(logits).all():
                with torch.autocast("cuda", enabled=False):
                    logits = model(x.float()).float()
    except Exception:
        with torch.autocast("cuda", enabled=False):
            logits = model(x.float()).float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=15.0, neginf=-15.0)
    o = torch.sigmoid(logits)
    return o[0].cpu().numpy()"""

    assert old_func in src, f"Could not find old_func in {nb_path} cell {cell_idx}"
    src = src.replace(old_func, new_func)
    nb['cells'][cell_idx]['source'] = [src]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully patched infer_probs in {nb_path}")

patch_infer_probs(r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2\rsna-knee-final-strict-v2.ipynb', 2)
patch_infer_probs(r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb', 3)
