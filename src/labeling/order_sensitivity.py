"""Robustness check of the locked v3.1 labeler: reverse the order in which the 12
items are answered (greedy feedback means earlier answers condition later ones).

Runs on dev (60 non-gold) + gold (58) studies only. Reported as a sensitivity
analysis; it is NOT used to change the locked method.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Coding/kaggle/RSNA/src/labeling")
import llm_assertion_labeler as L  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
tr = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
dev = pd.read_csv(f"{ROOT}/data/labels_v3/dev_pilot_uids.csv").StudyInstanceUID.tolist()
gold = pd.read_csv(f"{ROOT}/data/labels_v3/gold_uids_LOCKED.csv").StudyInstanceUID.tolist()
uids = dev + gold

orig_keys, orig_targets = list(L.ITEM_KEYS), list(L.TARGETS)
L.ITEM_KEYS = orig_keys[::-1]       # layout + forced order reversed; definitions in the system prompt unchanged
tok, model = L.load_model("Qwen/Qwen3-4B-Instruct-2507", "nf4", "D:/rsna_work/hf_cache/hub")
sc = L.Scorer(tok, model, prefix_cache=True)
enc = [sc.prompt_ids(tr.loc[u, "Report"])[0] for u in uids]
lengths = np.array([len(e) + len(sc.prefix_ids) for e in enc])
rows = []
for idx in L.make_batches(lengths, 8000, 4):
    probs, mass = sc.score_batch([enc[i] for i in idx])
    for j, i in enumerate(idx):
        r = {"StudyInstanceUID": uids[i]}
        for k_rev, key in enumerate(L.ITEM_KEYS):
            t = orig_targets[orig_keys.index(key)]
            for li, Lt in enumerate(L.LETTERS):
                r[f"{t}__p{Lt}"] = float(probs[j, k_rev, li])
        rows.append(r)
pd.DataFrame(rows).to_csv(f"{ROOT}/data/labels_v3/llm_v3.1_reversed_order_dev_gold.csv", index=False)
print("done", len(rows))
