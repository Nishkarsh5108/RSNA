# Repository data-exposure remediation plan (Option A done by the owner on 2026-10-19; Option B not needed)

**Current status (2026-10-19 11:21 IST):**
- **The owner made the repository private** this morning. Verified: the anonymous GitHub API now returns 404 for it.
- **History left as is (owner's decision).** The owner asked for the data to be purged from history only if the rules require that for a private repo. They do not:
  - Rule 4.b.1 forbids making the data available "to any party not participating in the Competition".
  - The MIRA license (rule 4.a.2) allows "a copy of a MIRA dataset for research use" and forbids distributing or publishing a copy "to others" without RSNA's written permission.
  - A private repo that only the owner can access is the owner's own research copy.
- **What keeps this true:**
  - No collaborator on the repo who has not accepted the competition rules.
  - Never make the repo public again while the data is in its history; run Option B first.
- **Still the owner's call:**
  - Rule 4.b.1 also says participants "notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data" and "work with Kaggle to rectify" it. The data was in every commit from the first one (2026-08-30) until the repo went private on 2026-10-19.
  - Forks made while the repo was public stay public after the original goes private. Check the fork count on the repo page.

**Earlier status (verified 2026-10-17 10:22 IST):** `https://github.com/Nishkarsh5108/RSNA` loads without sign-in and shows "Public".

**Re-verified 2026-10-19 04:37 IST:** still public (GitHub API: `"private": false`; last push 2026-09-13). The tracked files are unchanged (raw DICOM, `test.csv`, series tables and report-derived label CSVs are still in `git ls-files`). Option A below is still the recommended first step.

Git tracks competition data:
- `data/main_data/train (1).csv`: the official `train.csv`, with all 4,407 reports and the 58 gold labels;
- `data/main_data/train_series.csv`, `test*.csv`, and label files that embed the `Report` column;
- `1.2.826.0.1.3680043.8.498.11421887690184506354786744807184755984.dcm`: a raw competition DICOM at the repo root.

The competition rules forbid providing competition data to non-participants.

Since the first audit, nothing new has been committed or pushed. All new outputs that derive from competition data are git-ignored: `data/labels_v3/`, `data/imaging_v5/`, `reports/eda/`, `output/`.

## Option A — fastest risk reduction (minutes, reversible)
1. GitHub → repository Settings → General → Danger Zone → **Change visibility → Private**.
2. Check that the repo no longer loads in a private browser window.

## Option B — remove the data from history (irreversible for collaborators; do after A)
Run from a **fresh mirror clone** after taking a backup:

```bash
git clone --mirror https://github.com/Nishkarsh5108/RSNA.git rsna-mirror.git
cd rsna-mirror.git
pip install git-filter-repo
git filter-repo --invert-paths \
  --path data/main_data \
  --path 1.2.826.0.1.3680043.8.498.11421887690184506354786744807184755984.dcm \
  --path output/v1/submission.csv --path output/v1/submission_v1.csv
git push --force --mirror
```

The repo is private now, so the clone needs the owner's GitHub sign-in. The scope should also cover the saved outputs of `src/EDA/eda_qwen_labels.ipynb` (cell 3: a 5-row preview with truncated report text); its code can stay.

Then, in the working copy: `git fetch`, then `git reset origin/main`. This plain (mixed) reset moves the branch and the index and leaves every file on disk; the former data files become untracked. Add `data/main_data/` to `.gitignore`.

**Do not use `git reset --hard origin/main` here** (corrected 2026-10-19; an earlier version of this plan said to). A hard reset deletes every formerly tracked file from the disk, including all of `data/main_data/` and the DICOM, and discards uncommitted edits such as the current `.gitignore` change.

Caveats:
- Forks or clones made while the repo was public keep the data.
- GitHub may cache old views. Contact GitHub Support to purge cached views if needed.

## Not done by the agent
The owner changed the visibility (Option A). No force-push or history rewrite was performed: under the owner's instruction, a purge was needed only if the rules required it for a private repo, and they do not.
