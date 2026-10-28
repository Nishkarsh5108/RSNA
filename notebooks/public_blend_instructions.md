# Optional: blending our models into a public notebook (owner's decision)

**Read first.**
- Every public notebook scoring ≥ 0.950 (e.g. goodpjw2008 "CoAtNet + R6 + 2.5D MIL", 0.958) uses `nartaa/rsna-knee-publication-swa-weights-20261007`. That is a CoAtNet trained with **external OAI knee MRIs**.
- OAI is controlled-access data (NIMH Data Archive, data use certification). The hosts had not ruled on its use (discussion 743416) in any source read on 2026-10-18.
- A submission built on those weights carries that rules risk.
- **Clean option:** a public notebook whose weights come only from competition data (e.g. the Raptor CoAtNet family + goodpjw2008's 2.5D reader). Third parties reported about 0.945 for such blends.
- **Final picks:** keep one of the two final selections clean.

## How (no change to the public notebook's own code)

1. Fork ("Copy & Edit") the chosen public notebook in the competition.
2. **Add Input** → our private dataset (e.g. `rsna-knee-image-model-v3`). It contains the `knee_*.pt` files, `wheels/` and `rsna_knee_infer_standalone.py`.
3. Insert this as the **first** cell. It runs our models in a separate process, with their own package copies in `/tmp/knee_site`, so the public notebook's packages and GPU memory are untouched:

```python
import glob, subprocess, sys
script = glob.glob("/kaggle/input/**/rsna_knee_infer_standalone.py", recursive=True)[0]
subprocess.run([sys.executable, script, "--out", "/kaggle/working/submission_ours.csv"], check=True)
```

4. Add this as the **last** cell. It is a per-finding rank average; AUC depends only on ranks:

```python
import pandas as pd
W_OURS = 0.20                     # weight of our models; 0 = public notebook unchanged
pub = pd.read_csv("/kaggle/working/submission.csv")
ours = pd.read_csv("/kaggle/working/submission_ours.csv").set_index("StudyInstanceUID").loc[pub.StudyInstanceUID].reset_index()
cols = [c for c in pub.columns if c != "StudyInstanceUID"]
assert list(ours.columns) == list(pub.columns) and len(ours) == len(pub)
out = pub.copy()
for c in cols:
    out[c] = (1 - W_OURS) * pub[c].rank(pct=True) + W_OURS * ours[c].rank(pct=True)
out.to_csv("/kaggle/working/submission.csv", index=False)
print(out.shape, out[cols].min().min(), out[cols].max().max())
```

5. Accelerator and internet as the public notebook requires (internet **off** for submission). Then Save Version → Submit.

## Weight choice

Third-party experience:
- A 0.92-level own model blended at w = 0.10 into a 0.95 public stack changed the LB by −0.001 to +0.000.
- Only models that are both strong and differently built added value.

Suggested probes, one submission each:
1. The public notebook unchanged: the reference.
2. w = 0.20 with our best release.

Keep the blend only if it beats the reference by ≥ 0.002 on the public LB. Private-LB gains from smaller differences are not reliable.

## Runtime

Our release adds about 40–60 min (estimate for 1,300 studies on a T4) on top of the public notebook's own runtime.

Check that the total stays well under 9 h: Kaggle shows a submission's scoring status but not the full timing of its hidden-test run.
