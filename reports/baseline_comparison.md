# Baseline comparison: do better pseudo-labels train a better image model?

Questions addressed (mission §7): C, whether generated labels are useful for training an image model, and D, whether the vision model generalises to unseen studies. Results come from `src/labeling/downstream_eval.py` and `src/labeling/gold_only_baseline.py`. Raw outputs are in `reports/eda/downstream/`.

## 1. Audit of the existing baselines (v1–v4 notebooks)

| Aspect | v1 | v2 (best LB 0.775) | v3 | v4 (untracked) | Assessment |
|---|---|---|---|---|---|
| Input | 3rd-party `.npz` (24×224×224) | own DICOM → `.npy` (T×224×224, uint8, 1–99% window) | same + Laterality-tag mirroring (cor/ax) | v3 preprocessing | Slice ordering by IPP·normal is **correct** (verified: 0 non-monotonic series in a 2,000-series sample). v1's third-party cache lacked the hidden test studies, giving LB 0.500. |
| Series selection | 1 per plane (fluid-sensitive first) | same | up to 3 per plane + mask | same | Reasonable. Fluid-sensitive is missing for 257 sagittal / 159 coronal studies. |
| Slices | 12 uniformly sampled by index | 12 | 36 slots (3×12) | 12 / 8 | Index sampling ignores physical coverage (3D series up to 320 slices) |
| Backbone | EfficientNet-B0 ImageNet, **frozen** | frozen | frozen | DINOv2-small frozen (fine-tune optional) | **Frozen ImageNet features are the main performance limiter** |
| Head | attention-pool per plane → MLP | same | region-guided plane attention | same | Fine |
| Labels | consensus | consensus | consensus + disagreement weights | consensus | The consensus contains a corrupted voter (`public_pseudo` = 0.5 everywhere). Downstream, the effect is small (§3). |
| Gold use | gold rows in training folds with ×8 weight | same | same | same | Acceptable for training, but contaminates any gold evaluation of those checkpoints |
| Split | `StratifiedGroupKFold` grouped by StudyInstanceUID | same | same | same | Equivalent to study-level stratified K-fold. Patient grouping is impossible (IDs per study). The 1 duplicate pair is not forced into one fold. |
| Checkpoint selection | **best epoch by gold AUC** on that fold's gold studies | same | same | same | **Selection bias.** The reported "honest gold AUC" is a maximum over 20 noisy evaluations on 12–15 studies. |
| Reported gold AUC | 0.7538 (fold 0) | 0.7601 (fold 0, 13 gold) / 0.7259 (5-fold, per v4 note) | 0.7294 (fold 0) | 0.7552 (5-fold OOF, n=58) | All optimistic (selection on gold). Fold-0-only numbers rest on 13 studies. Per-class values such as "Lateral OA 1.000" rest on 1–3 positives. |
| Loss | BCE + pos_weight (clip 1–10) + gold ×8 | same | + disagreement weights | same | Fine for soft labels |
| Metric code | macro AUC skipping single-class columns | same | same | same | Correct |
| Thresholds | none (AUC) | none | none | none | Correct for this metric |
| Augmentation | none (frozen cache) | none | none | intensity jitter (fine-tune path) | Little regularisation possible with frozen features |
| Reproducibility | seeds set | seeds set | seeds set | seeds set | Results depend on gold-based selection |

## 2. Valid evaluation protocol used here

- **Features.** The v2 frozen EfficientNet-B0 cache (exactly the representation behind LB 0.775), so differences isolate the **label source**.
- **Model.** The v2 head, retrained from scratch for every label set: AdamW (lr 1e-3, wd 1e-4), batch 64, BCE with `pos_weight` clipped to [1, 10], NaN targets masked.
- **Training data.** Only the 4,349 non-gold studies. **Gold is never used** for training, early stopping, checkpoint selection or thresholds.
- **Epochs.** 12 for every run, fixed before any result was seen.
- **Seeds.** 5 per label set; the reported AUC is computed on the seed-averaged predictions. The per-seed mean ± SD is also reported.
- **Evaluation.** All 58 gold studies (image-read labels). Macro ROC-AUC with study-bootstrap 95% CI (2,000 draws). Label sets are compared with a **paired** bootstrap on the same studies.
- **Caveat.** n = 58 means a macro-AUC SE of about 0.025. Differences below about 0.03 are not resolvable, and the comparisons below are interpreted accordingly.

## 3. Results: existing label sources

| Label set used for training | Gold macro AUC (seed-avg) [95% CI] | Per-seed mean ± SD |
|---|---|---|
| Qwen2.5-7B (original pipeline) | 0.751 [0.696, 0.804] | 0.741 ± 0.010 |
| Qwen2.5-14B-AWQ | 0.731 [0.679, 0.782] | 0.721 ± 0.010 |
| gpt56sol (public LLM set) | 0.770 [0.723, 0.818] | 0.761 ± 0.006 |
| report_v2 (public verdict scorer) | 0.770 [0.712, 0.820] | 0.757 ± 0.003 |
| consensus (current training labels) | 0.775 [0.726, 0.824] | 0.764 ± 0.004 |
| mean(gpt56sol, report_v2) | 0.781 [0.732, 0.827] | 0.767 ± 0.004 |
| **Human labels only** (58 gold, leave-one-out; PCA16 + L2-LR, C fixed a priori) | 0.590 [0.521, 0.661] | — |
| Human labels only (PCA8) | 0.506 [0.450, 0.559] | — |

Observations:
1. **Report-derived pseudo-labels are essential.** Every pseudo-label set beats training on the 58 gold studies alone by more than 0.14 AUC. Caveat: the gold-only model is necessarily a smaller model class, since a 1M-parameter head cannot be trained on 57 studies.
2. The original Qwen-7B/14B labels train slightly worse image models than the public sets or the consensus. The 0.02–0.05 gaps are at or below the resolution limit, but they line up with their lower gold agreement (`label_validation_report.md`).
3. Averaging two strong, partly independent sources gives the best result among the existing sets.

## 4. Results: new label sets (labels_v3), all re-run together on GPU

Same protocol, 5 seeds, `reports/eda/downstream/gpu_all/`. Reference sets were re-run in the same session for like-for-like pairing. **GPU vs CPU training of the same label set differs by about 0.006–0.010** (e.g. mean(gpt56sol, report_v2): 0.781 CPU vs 0.775 GPU). That is the noise floor of this protocol.

| Label set used for training | Masked cells | Gold macro AUC (seed-avg) [95% CI] | Per-seed mean ± SD | Δ vs mean(gpt56sol, report_v2) [95% CI] |
|---|---|---|---|---|
| **ens(LLM + gpt56sol + report_v2)** | 0% | **0.784** [0.739, 0.829] | 0.770 ± 0.007 | +0.009 [−0.003, 0.024] (94% > 0) |
| LLM, gold-calibrated mapping (**gold-informed, flagged**) | 0% | 0.779 [0.735, 0.824] | 0.765 ± 0.006 | +0.005 [−0.020, 0.029] |
| ens(all 4 incl. rules) | 0% | 0.775 [0.728, 0.821] | 0.766 ± 0.009 | +0.000 [−0.012, 0.011] |
| mean(gpt56sol, report_v2) (reference) | 0% | 0.775 [0.727, 0.822] | 0.767 ± 0.010 | — |
| LLM hard (A=1, B=0.5, E=0.25, C/D=0) | 0% | 0.772 [0.725, 0.816] | 0.750 ± 0.010 | −0.003 [−0.031, 0.022] |
| consensus (current training labels) | 0% | 0.772 [0.725, 0.818] | 0.763 ± 0.011 | −0.003 [−0.021, 0.014] |
| ens(LLM + public), silent cells masked | 14.1% | 0.770 [0.721, 0.816] | 0.757 ± 0.008 | −0.005 [−0.024, 0.014] |
| LLM v3.1 soft (a priori s0) | 0% | 0.768 [0.721, 0.813] | 0.749 ± 0.009 | −0.007 [−0.031, 0.017] |
| gpt56sol | 0% | 0.765 [0.716, 0.814] | 0.753 ± 0.012 | −0.010 [−0.030, 0.010] |
| report_v2 | 0% | 0.764 [0.709, 0.813] | 0.752 ± 0.004 | −0.011 [−0.029, 0.007] |
| Qwen2.5-7B (old) | 0% | 0.761 [0.709, 0.812] | 0.741 ± 0.009 | −0.014 [−0.041, 0.014] |
| ens(all 4), silent cells masked | 14.1% | 0.758 [0.707, 0.804] | 0.749 ± 0.009 | −0.017 [−0.039, 0.003] (95% < 0) |
| LLM s0, "not mentioned" masked | 15.3% | 0.757 [0.706, 0.805] | 0.744 ± 0.007 | −0.018 [−0.042, 0.005] (94% < 0) |
| rules v1.1 | 0% | 0.747 [0.695, 0.798] | 0.733 ± 0.010 | **−0.028 [−0.053, −0.003]** |

**Order-averaged LLM (final).** Same GPU protocol, `reports/eda/downstream/final/`. Re-run reference sets reproduced their earlier GPU values exactly (0.7748, 0.7842), so the GPU protocol is deterministic.

| Label set | Gold macro AUC [95% CI] | Δ vs mean(gpt56sol, report_v2) |
|---|---|---|
| ens(LLM canonical + gpt56sol + report_v2) | 0.784 [0.739, 0.829] | +0.009 [−0.003, 0.024] |
| **final primary: ens(LLM order-avg + gpt56sol + report_v2)** | **0.782** [0.735, 0.827] | +0.007 [−0.004, 0.020] |
| LLM order-averaged alone | 0.767 [0.723, 0.812] | −0.008 [−0.030, 0.016] |
| LLM canonical alone | 0.768 [0.721, 0.813] | −0.007 [−0.031, 0.017] |

Order averaging improves gold *agreement* (+0.012) but does not change downstream image-model AUC (−0.001/−0.002). With frozen features, label differences of this size are below what the downstream protocol can detect.

## 5. Answers to the mission's questions (§7.4)

| # | Question | Answer (evidence level) |
|---|---|---|
| 1 | Are the current labels usable? | **Yes, with corrections.** The consensus trains an image model as well as any single public source (0.772–0.775), even though it contains a constant-0.5 voter and gold-contaminated weights. The `public_pseudo` file itself is unusable (no information). The old Qwen files are usable but weaker (gold 0.80; downstream −0.014 to −0.024 vs the consensus). [measured] |
| 2 | Most reliable classes | ACL, MCL, Medial meniscus, Medial OA, Baker's: gold AUC ≥ 0.92 for the final ensemble. MCL and Baker's rest on only 9 and 12 positives. [measured] |
| 3 | Least reliable classes | **Synovitis (0.74)**, then Lateral OA (0.82, n+ = 11), Contusion (0.85), Fracture (0.85), Effusion (0.87). For Synovitis, Effusion and Fracture the main cause is report/image discordance, not extraction. [measured] |
| 4 | Problem languages | Greek and German have the highest cell error for every labeller (2–3 gold studies each, so not statistically resolvable). Greek text was 95% micro-sign corrupted and is now normalised. [measured; weak] |
| 5 | Does the proposed method improve on the existing one? | **Over the previous local pipeline (Qwen2.5 + 1,200-char truncation): yes.** Gold +0.055 [0.012, 0.093]; downstream directionally better. **Over the best public label sets: no as a standalone source** (it sits between gpt56sol and report_v2). **As an ensemble member it is complementary**: highest gold agreement (0.884) and best downstream result (0.784), though neither gain over the public-only ensemble is statistically resolved. [measured] |
| 6 | Does adding pseudo-labelled data help the vision baseline? | **Yes, decisively.** Gold-only training: 0.53–0.60 (repeated 5-fold; LOO 0.51–0.59). Pseudo-label training: 0.75–0.78. [measured; model classes differ by necessity] |
| 7 | Is the improvement consistent across classes? | Mostly. For gold agreement, the final ensemble beats the old Qwen-7B labels on 11/12 classes (largest gains: MCL +0.22, PF OA +0.15, Effusion +0.14) and ties on Synovitis (0.740 vs 0.743). Downstream per-class effects are within noise (`downstream_results.csv`). [measured] |
| 8 | Supported by a valid validation design? | Gold is never used for training or selection, epochs are fixed, seeds are averaged and comparisons are paired. **But** n = 58 makes differences below about 0.03 unresolvable. Some choices (primary-ensemble membership, order averaging) were informed by gold and are logged as post-hoc. The old notebooks' "honest gold AUC" is **not** valid (best-epoch selection on gold). [verified] |
| 9 | Remaining limitations | Frozen ImageNet features cap the image model at about 0.78 gold AUC, so label effects are hard to see. Gold is image-read, tiny and enriched. Public label-set provenance is unknown. The local 4B model is prompt- and order-sensitive. [measured/claim] |
| 10 | Which label strategy next? | **Train on the soft mean of {order-averaged local LLM, gpt56sol, report_v2}** (`training_labels_v3.csv`; gold agreement 0.890, downstream 0.782, statistically tied with the best alternatives). Keep "not mentioned" as a low soft score (don't mask). Use soft BCE. Never use gold for selection. Spend the remaining effort on the **image model** (fine-tuned 2.5D, laterality normalisation, decoder robustness), where the public evidence suggests gains of +0.1 AUC versus about +0.01 from label changes. [recommendation] |

## 6. What was not done

- No fine-tuned (unfrozen) 2.5D image model was trained. The downstream test isolates label effects on a fixed, frozen representation, so label-quality effects on a stronger model remain an untested hypothesis.
- Leaderboard submissions were not made, and no Kaggle runs were executed from here (no credentials).
- The v1/v3 checkpoints in `output/` were not re-evaluated. Their gold evaluation would be contaminated anyway, because gold rows were in their training folds.
