# How to submit an emergency release (one leaderboard test)

Every release is a folder under `D:/rsna_work/kaggle_upload/<release-name>/`. It contains:
- the `knee_*.pt` checkpoints (each carries its preprocessing, ensemble weight and window settings);
- `wheels/`;
- `rsna_knee_infer_standalone.py`;
- `README.md` and `SHA256SUMS.txt`.

The 0.927 release (`rsna-knee-image-model`) and its notebook are never modified. Keep that notebook version as fallback **C**.

## One-time setup

1. Kaggle → competition → **Code → New Notebook** → File → **Import Notebook** → `notebooks/rsna_knee_image_model_inference_emergency.ipynb`.
2. Settings: **Accelerator GPU T4 ×1**, **Internet off**.

## Per release

1. Kaggle → **Datasets → New Dataset**:
   - upload the *contents* of the release folder, keeping `wheels/` as a sub-folder;
   - set visibility to **Private**;
   - any name, e.g. the release name.
2. In the emergency notebook → **Add Input** → that dataset. **Remove** any other dataset that contains `knee_*.pt` files: the notebook uses the first folder it finds with `knee_*.pt`.
3. **Run All** (dry run on the 3 public test studies). The dependency cell should show all four decoders as `True`, and the last cell should print `submission OK`.
4. **Save Version** (Save & Run All) → **Submit** that version. The hidden test takes roughly 30–90 min; the local estimate per release is in its README.
5. Report the public LB score back. If you can see it, also report the scoring time.

## Several submissions from one upload

You can upload one dataset containing several checkpoints, then choose the subset and weights per submission. Change this one line in the CONFIG cell before Save Version:

```python
SELECT_MODELS = None                                    # every knee_*.pt in the folder, packaged weights
SELECT_MODELS = {"knee_F1_cnxtT_s3_k8_attn12_pf_all_s42.pt": 1}          # e.g. one model only
SELECT_MODELS = {"knee_F1_cnxtT_s3_k8_attn12_pf_all_s42.pt": 1, "knee_E9_hr_cnxtT_s3_k6_attn12_c320_pf.pt": 1}
```

- Names must match the files exactly. An unknown name stops the run with an error, so it can never silently use a different set.
- The weights are relative.

## Notes

- **Paths:** auto-discovered. If discovery fails, set `MODEL_DIR = "/kaggle/input/<dataset-name>"` in the CONFIG cell.
- **Packages:** if the decoder line shows `False`, the `wheels/` folder was not uploaded.
- **Do not** upload competition data, labels or reports. Releases contain model weights and code only.

## Monday 2026-10-19: public base (P0) and the single-model tests (S2)

Kaggle's daily limit of 5 submissions resets at 00:00 UTC (05:30 IST).

### P0: the public OAI-trained notebook (main-pick candidate)

Pick one base (owner's decision; see `reports/emergency_experiment_plan.md` §10b):

| Base | Public LB | External data |
|---|---|---|
| **A-strict**: heliosli `rsna-knee-blend-gold-dataset-restored` (nartaa 384 + S6, 70/30) | 0.954 | OAI + SKM-TEA; no fastMRI marker |
| **A-R6**: goodpjw2008 `rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` v27 | 0.958 | OAI + SKM-TEA; R6 may be fastMRI-trained (unverifiable) |

1. Open the notebook in the browser → **Copy & Edit**. This keeps its inputs at the exact versions it scored with; the API cannot do that.
2. A-R6 only: in the input panel, **remove** "RSNA Knee Compact R7 CNX FS SS". No code cell loads it (checked), and removing it keeps R7 out of the submission.
3. Settings: **GPU T4 ×2**, **Internet off**. Leave everything else unchanged.
4. **Save Version** (Save & Run All; about 10 min on the 3-study stub) → when it finishes, **Submit**. The hidden run takes several hours.

### S2a–S2d: our single models (decide the clean second pick)

| # | Dataset | `SELECT_MODELS` |
|---|---|---|
| S2a | S1 dataset (already uploaded) | `{"knee_F1_cnxtT_s3_k8_attn12_pf_all_s42.pt": 1}` |
| S2b | `rsna-knee-v3-all` (new upload) | `{"knee_S224_PLg_all_s42.pt": 1}` |
| S2c | `rsna-knee-v3-all` | `{"knee_SHR_rep_all_s42.pt": 1}` |
| S2d | `rsna-knee-v3-all` | `{"knee_SHR_PLg_all_s42.pt": 1}` |

Rules (plan §9): PL is adopted only if S2b ≥ S2a + 0.003 **and** S2d ≥ S2c + 0.003. HR loses its ensemble role only if S2c ≤ S2a − 0.005.

### P1 (later, optional): our ensemble at a 0.2 share inside P0

Paste `notebooks/part_c_ours_blend_cell.py` as the last cell of the P0 fork, attach our release dataset, Save Version → Submit. It becomes the main pick only if it beats P0 by ≥ 0.002 (plan §10a). It skips itself and keeps P0's submission unchanged if time or disk run short.

## Revised Monday plan (2026-10-19 04:25 IST; supersedes the section above)

Plan §10c records the evidence behind these changes.
- **A-R6 is no longer recommended.** R6's own trainer adds a fastMRI+ auxiliary task, and R6 was trained from a run named `cnx_r6fm`. fastMRI's application requires an institutional affiliation.
- **P1 is dropped.** The base takes 6.5–8 h, and a 0.93 leg adds nothing at a 20 % share.
- **S2a–S2d no longer feed any final pick.** Run them only if slots are left.

### Submissions (2 slots)

| # | Notebook | Inputs (all public) | Expected |
|---|---|---|---|
| A1 | **A-strict**: heliosli's `rsna-knee-blend-gold-dataset-restored`, unchanged | S6, nartaa SWA weights, orthofoundation (`anatomy.pt`) | 0.954 (displayed on its page) |
| A2 | **A-strict+**: v27 with R6 → S6 (`notebooks/candidate_A/a_strict_plus/`, built by `src/emergency/build_a_strict_plus.py`) | the above + goodpjw2008's reader | 0.954–0.955 (hypothesis; most likely 0.954) |

**Route 1, API (needs the owner's go-ahead; nothing has been uploaded):**

```bash
kaggle kernels push -p notebooks/candidate_A/a_strict        # creates private nishkarshsinghal/rsna-knee-a-strict, runs it on the 3-study stub
kaggle kernels push -p notebooks/candidate_A/a_strict_plus   # creates private nishkarshsinghal/rsna-knee-a-strict-plus
kaggle kernels status nishkarshsinghal/rsna-knee-a-strict-plus   # wait for "complete"; then read the log for NOTEBOOK_COMPLETE
kaggle competitions submit rsna-knee-abnormality-detection -k nishkarshsinghal/rsna-knee-a-strict -v 1 -f submission.csv -m "A-strict: nartaa + S6 (heliosli recipe)"
kaggle competitions submit rsna-knee-abnormality-detection -k nishkarshsinghal/rsna-knee-a-strict-plus -v 1 -f submission.csv -m "A-strict+: v27 with S6 in place of R6"
```

**Done on 2026-10-19 via route 1, with the owner's approval:** both pushed as version 1 at 11:09 IST; both commit runs passed every check below; both submitted at 11:19 IST (submissions 57384567 and 57384570). Do not submit them again; a rerun only spends a slot.

**Scored (seen 2026-10-19 17:32 IST):** A1 A-strict **0.954**, A2 A-strict+ **0.954**. By the pick rule below, A2 is the main pick and A1 the second. Both are to be ticked as final.

**Superseded on 2026-10-20 (seen 05:37 IST):** final-strict (`nishkarshsinghal/rsna-knee-final-strict` v1, submission 57406523, submitted 00:35 IST with the owner's approval) scored **0.956**. By the plan §11 rule, **tick final-strict + A-strict+ as the two final picks** by Thu 22 Oct 23:59 UTC. A-strict is the fallback. Do not resubmit any of the three: a rerun only spends a slot and GPU time.

Both metadata files pin heliosli's 0.954 Docker image (the image the S6 stage ran on), T4 ×2 and internet off. New slugs never touch the existing notebooks: `notebook45cfc61c19` is the 0.927 fallback C, `notebook535502ec96` is S1.

**Route 2, browser:**
- **A-strict:** open the notebook → **Copy & Edit** → GPU T4 ×2, Internet off → **Save Version** → **Submit**.
- **A-strict+:** Create → New Notebook → File → Import Notebook (the `.ipynb`). Add the four datasets listed in its `kernel-metadata.json`, set GPU T4 ×2 and Internet off, then Save Version → Submit. This route cannot pin the Docker image; route 1 can.

### Commit-run checks (the 3-study stub, about 10 min)

The log must show:
- `RAPTOR_STAGE_DONE`
- `R224_STAGE rc=0 ok=True` (A-strict+ only)
- `ASSETS_READY` and `OURS_STAGE_COMPLETE`
- `NOTEBOOK_COMPLETE`
- `own leg: blended 3 checkpoints` (A-strict+ only; `own leg FAILED` or `Part B skipped` means the reader did not run)

Any traceback means the version is not submitted.

### Pick rule (plan §10c, fixed now)

A-strict+ is the main final pick if it completes and displays ≥ A-strict's score; A-strict is the second pick. If A-strict+ scores lower, the order reverses. Our own 0.927/0.928 notebooks stay in the history as fallback C; they are not final picks.

### Final selection (added 2026-10-19 07:24 IST; deadline Thu 22 Oct 23:59 UTC = Fri 05:29 IST)

- Submissions tab → tick the two A variants as the final submissions.
- If nothing is ticked, Kaggle normally selects the best public scores automatically. **Do not submit any R6 or R7 notebook** (the 0.956–0.958 ones), not even as a test. It would outscore both A variants and could end up selected.
