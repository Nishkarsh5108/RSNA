# Handover: RSNA knee post-competition work (state at 14 Nov 2026, 23:25 IST)

Read first: `reports/POSTCOMP_DECISION_LOG.md` (every decision P1-P41, run register, evidence) and the memory folder
`C:\Users\nishk\.claude\projects\d--Coding-kaggle-RSNA\memory\` (MEMORY.md index).

## Goal and deadline
Owner presents results on **17 Nov 2026**. Target: match the winners (private 0.960; ranks 3-9 at 0.957-0.959).
Competition-period best was private 0.943 (rank 490). Competition is closed: late submissions show on the account but cannot change the rank.

## Owner instructions in force
- **Submissions and inference notebooks: owner's Kaggle account** (user `nishkarshsinghal`, token file `C:\Users\nishk\.kaggle\kaggle API.txt`).
- **Training: friend's account** (`anujchaubey`, token `C:\Users\nishk\.kaggle\kaggle-api_friend.txt`); quota may be exhausted.
- Resubmit the best result to the owner's account (done for A2 and A2+C1).
- Log every decision in the decision log as it happens.
- Never print tokens (pass via `KAGGLE_API_TOKEN` in the child process only). Notebooks/submissions must not mention Claude or any tool.
- Do not commit competition-derived data to git. Gold-58 is never trained on and never used to select models.
- Kaggle CLI (`D:\rsna_work\kaggle_cli_venv\Scripts\kaggle.exe`): use `D:/...` paths, not Git-Bash `/d/...`.

## What was built
Base model = 1st-place public box-pooling classifier (tascj0), trained on the friend's 2xT4 (~6 h per run, 8 epochs, ConvNeXt-T 256 px).
Our contribution: labels (7th-place + 1st-place label sets, averaged = "fused"), the inference pipeline, the scripts, the tests.
Scripts in `src/postcomp/`: `build_bp_kernel.py` (training notebook builder), `build_infer_kernel.py` + `infer_bp.py` (inference notebooks),
`kaggle_flow.py` (push, wait, submit once, state file, blocks idle sleep), `post_training.py` (folds 2/3 -> weights dataset -> final ensembles),
`score_bp.py`, `sweep_sharpen.py`, `series_weighting.py` (analysis). Generated notebooks: `D:/rsna_work/kaggle_runs/`.

## Results (private / public LB, late submissions)
| Model | Private | Public |
|---|---|---|
| A0 (1st labels, fold 0) | 0.955 | 0.963 |
| A1 (7th labels, fold 0) | 0.957 | 0.967 |
| A0+A1 | 0.958 | 0.967 |
| A2 (fused labels, fold 0) | 0.958 | 0.968 (friend's account and owner's account both) |
| A2+C1 (owner's account) | 0.958 | 0.968 |

Rejected (see log): own study-level model in ensemble (P27), higher resolution in our model (P17), attention sharpening (P33, best +0.0005),
series-type weighting (P34, best +0.0007), test-time augmentation (skipped, reasoning in chat). Label source barely matters in this model (P25, P39).
Differences between 0.955-0.958 are within test-set noise.

## Running / pending right now
- Friend's account: `knee-bp-c-f2`, `knee-bp-c-f3` (fused labels, folds 2 and 3) both RUNNING since ~23:10; expect done ~05:00-06:00 on 15 Nov.
- Local background process `post_training.py` (check `D:/rsna_work/post_training.log`; python PID was running at 23:25): waits for both runs
  (deadline Mon 16 Nov 05:00 IST, then proceeds with completed folds), downloads weights, adds `c2_f2.pth`, `c3_f3.pth` to the owner's private dataset
  `nishkarshsinghal/knee-bp-weights` (already holds `a0_f0, a1_f0, a2_f0, c1_f1`), builds and pushes on the owner's account, via `kaggle_flow.py`:
  `knee-bp-sub-f6` (**pre-registered primary**: equal-weight mean of all six models), `knee-bp-sub-f4` (ablation: four fused models), then research
  checks `knee-bp-sub-a1-sv`, `-a012`, `-a1-s2`, `-a012c1` (built already, in `D:/rsna_work/kaggle_runs/`). Log: `D:/rsna_work/flow_owner_final_state.log`.
- If `post_training.py` died (laptop sleep, restart): rerun `cd src/postcomp; python -u post_training.py` (state files make flows safe to restart; it checks training status first).
  Manual fallback: download outputs of `knee-bp-c-f2/-f3` (friend's key), copy `fold2_ema.pth`/`fold3_ema.pth` as `c2_f2.pth`/`c3_f3.pth` into
  `D:/rsna_work/kaggle_runs/knee-bp-weights-ds`, `kaggle datasets version -p <dir> -m ...` (owner's key), build with `build_infer_kernel.py --user nishkarshsinghal --weights-dataset nishkarshsinghal/knee-bp-weights --cls knee-bp-weights/<file>,...`, push, submit with `-k nishkarshsinghal/<slug> -v <version> -f submission.csv -m "<message>"`.
- Do not report the best-scoring variant as "the" result: the primary (six-model mean) was fixed before results; report all variants.

## Demo site + workstation for 17 Nov (built 15 Nov, decisions P42–P46)
- `demo/` — landing site (method, contributions, results) + `#/workstation` (drop a DICOM study → 3D in patient space → live
  GPU inference trace → findings with exact per-cell evidence). Run: `powershell -ExecutionPolicy Bypass -File demo\start.ps1`
  (serves http://127.0.0.1:8000). Details and limits: `demo/README.md`.
- Loads every `*.pth` in `D:/rsna_work/kaggle_runs/knee-bp-weights-ds` at server start: once `post_training.py` adds
  `c2_f2.pth`/`c3_f3.pth`, restart the server and the workstation runs the six-model ensemble.
- Fidelity: `python demo/server/tests/test_fidelity.py` (engine vs `infer_bp.py`, max |diff| 1.6e-4). Rerun after any engine change.

## Practical traps
- Laptop sleep freezes the polling scripts (keep plugged in, lid open). A finished background task does not wake an idle session.
- Kaggle GPU sessions can sit QUEUED for 1-4 h; QUEUED sessions count toward the 2-session limit; no CLI cancel (only `kernels delete`).
- `TaskStop` on a bash script kills only the parent; use single-process python scripts (one earlier incident, P40).
- Weekly friend quota 30 h (reset assumed Sat 05:30 IST, unverified); roughly 23 h used by folds 0-3 runs since reset.

## Still to do
1. When f6/f4 are scored: add scores to the decision log (P-rows, run register) and to memory (`rsna-knee-postcomp-research.md`).
2. Write the 17 Nov results summary (competition 0.943 -> 0.958 single/2-model -> final ensemble), with honest caveats above.
3. Optionally: commit `src/postcomp/` and the log when the owner asks (never data/weights).
