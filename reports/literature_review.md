# Literature review: report-derived labels for knee MRI with scarce gold labels

Scope: methods for turning free-text radiology reports into training labels when only a few studies have expert labels, and how those labels then train an image model. Written 2026-10-17 for the RSNA Knee Abnormality Detection competition.

Every row below is marked as one of:
- **[V]** verified: I read the primary source or checked the claim locally;
- **[A]** author claim: stated by the authors or participants and not independently checked;
- **[H]** hypothesis.

Leaderboard numbers from different validation schemes are **not comparable** and are listed only as context.

## 1. The central fact the literature predicts: report labels ≠ image labels

| Source | Finding | Relevance here |
|---|---|---|
| Jain et al., *VisualCheXbert* (ACM CHIL 2021; arXiv 2102.11467) [V, abstract] | Radiologists labeling **reports** disagree substantially with radiologists labeling the **images** of the same chest X-rays (average κ ≈ 0.31–0.43). Causes: findings left unreported, clinical history visible only to report readers, Findings vs Impression differences, and label noise. A report→image-label model improved F1 by 0.14 over a report labeler. | The 58 gold labels are, by several participant accounts, an independent **image** read [A]. So the ceiling for a "perfect report reader" against gold is well below 1. Local evidence: gold Synovitis prevalence is 46.6% while reports mention synovitis in only about 12% of studies; Fracture is 31% vs 6.7% (`reports/eda/reports_labels/prevalence_gold_vs_reports.csv`) [V]. |
| Jain et al., *Effect of Radiology Report Labeler Quality on Deep Learning Models for CXR* (arXiv 2104.00793, 2021) [V, abstract] | Better report labelers (CheXpert → CheXbert → VisualCheXbert) produce better image classifiers. | Supports evaluating label sources **downstream** (Question C), not only text-vs-gold agreement. |
| Bishnoi, HF blog (2026-08-09) [A] | Estimates a 0.88–0.92 ceiling for report-derived labels against gold, because reports and images disagree. | Consistent with the best public label sets reaching ≈0.87–0.89 macro AUC against gold (below). |

## 2. Report-labeling methods

| Method (source) | Label generation | Uncertainty / negation | Evidence of quality | Strengths | Weaknesses for this dataset | Reproducible / allowed here |
|---|---|---|---|---|---|---|
| NegEx (Chapman 2001, JBI); ConText (Harkema 2009, JBI) | Trigger-term rules for negation, hypothetical, historical and experiencer context | Rule scopes | Widely replicated on English clinical text | Transparent, fast, deterministic | English-centric; cue lists must be rewritten per language; post-nominal negation (Turkish, Greek) needs post-scope | Yes. Re-implemented for 9 languages here (`src/labeling/rule_extractor.py`) |
| NegBio (Peng et al., AMIA 2018) [V, abstract] | Universal-dependency patterns for negation/uncertainty scope | Explicit uncertainty class | +9.5% precision over NegEx on CXR reports | Better scope than token windows | Needs English parsers; no multilingual weights | Idea reused (scope); code not applicable |
| CheXpert labeler (Irvin et al., AAAI 2019) [V] | Mention extraction → classification → aggregation, with an explicit *uncertain* class and U-Zeros/U-Ones/U-Ignore policies | Yes | Policy choice matters per class (e.g. atelectasis F1 0.30 under U-Zeros vs 0.51 under U-Ones) | The uncertainty-policy idea transfers directly | Chest-specific vocabulary; English only | Concept reused: assertion categories A/B/C/D/E here |
| CheXbert (Smit et al., EMNLP 2020) [V] | BERT trained on rule labels, then fine-tuned on a small expert set with back-translation | 4-class (pos/neg/unc/blank) | Significantly beat the rule labeler | Rules + few gold → strong model | Needs hundreds of expert report annotations; English | Not feasible: only 58 image-read (not report-read) labels |
| RadGraph (Jain et al., NeurIPS D&B 2021) | Entity/relation graph extraction | Assertion via relation types | Strong on CXR | Rich structure (anatomy-observation links) | English, chest, needs a trained model; no knee schema | Not applicable without a knee-specific schema |
| Weak supervision / label models: Snorkel (Ratner et al., VLDB 2017); Dawid & Skene (1979); CROWDLAB (Goh, Tkachenko, Mueller 2022) | Combine several noisy labelers | Model-dependent | Works when labelers are numerous and conditionally independent | Uses many cheap sources | Sources here are **correlated** (gpt56sol vs report_v2 κ≈0.87) and few; the previous pipeline also fed a constant 0.5 placeholder source into CROWDLAB (see `label_validation_report.md`) | Allowed; must be validated out-of-fold |
| Local open LLM labelers: Vicuna-13B (Mukherjee et al., *Radiology* 2023, doi 10.1148/radiol.231147) [V, abstract]; Llama-2-70B structuring German/English reports (*Eur Radiol* 2024, s00330-024-11074-y) [V, abstract] | Zero-shot prompting on-premise | Prompt-defined | Comparable to rule labelers on CXR; German MCC 0.66 vs English 0.75 | Privacy-preserving; multilingual zero-shot | Lower accuracy in non-English; self-reported "confidence" is not calibrated; JSON parsing failures | Allowed (public weights, local). Used here: Qwen3-4B-Instruct-2507 with **forced-choice token probabilities** instead of free JSON |
| Hosted LLM APIs (GPT/Gemini/Claude) | Zero-shot prompting | Prompt-defined | Strongest single labelers in this competition's public sets | High quality | **Data-security clause** (no providing competition data to third parties) makes this a compliance risk | Not used here |

## 3. Knee-MRI image models (context for the downstream use of labels)

| Source | Data / labels | Architecture | Reported metric | Notes |
|---|---|---|---|---|
| MRNet (Bien et al., *PLOS Med* 2018) | 1,370 exams; report-derived labels (abnormal, ACL, meniscus) | AlexNet per slice, max-pool over slices, per-plane logistic fusion | AUC 0.937 abnormal / 0.965 ACL / 0.847 meniscus (internal test) | The "2D backbone + slice pooling" MIL template that most public solutions here follow |
| fastMRI+ (Zhao et al., *Sci Data* 2022) | 1,172 knee exams with bounding boxes and study labels incl. MCL sprain grades | n/a (dataset) | n/a | Possible external pretraining/validation source (licence check needed) |
| KneeMRI (Štajduhar et al. 2017) | 917 sagittal PD exams, ACL grades | n/a | n/a | Small, single plane |
| Attention-based deep MIL (Ilse et al., ICML 2018) | n/a | Gated attention pooling | n/a | Basis of the "attention pool" used in v1–v3 here |

## 4. Public solutions for THIS competition (2026)

| Source (date) | Label strategy | Image model | Validation | Reported score | Verifiable? | Allowed? | Lesson |
|---|---|---|---|---|---|---|---|
| Y. Bishnoi, HF blog (2026-08-09) | Qwen3.6-35B (local, Ollama, temp 0) with a prompt revised against gold (0.833 mean accuracy, 0.766 F1 on gold); fused with a public label set; per-class accuracy-weighted votes | EfficientNet-B3, single channel, MIL with max pooling, up to 3 series × 12 slices, 288 px | GroupKFold on studies; found and fixed a gold leak (0.936 → 0.857 honest gold AUC) | Public LB 0.903, rank 18/792 (early in the competition) | [A] (blog) | Yes | Gold leakage is easy to introduce; max pooling beat attention; prompt tuning on gold inflates gold accuracy |
| aditbytes/kaggle-rsna-knee-abnormality (snapshot 2026-10-06) | Public LLM label sets; best ≈0.89 macro AUC vs gold; Synovitis weakest (≤0.79) | EfficientNetV2-S 2.5D (3-slice windows), transformer over windows, per-finding attention; canonical reorientation, 0.6 mm resampling | 58 gold **never trained on**, used as validation; 4 folds | Gold 0.902 (v3, all folds); public LB 0.923 alone, 0.943 blended | [A] | Yes | Gold-held-out validation tracks the LB; most top teams fork a 0.943–0.944 public stack |
| goodpjw2008 "2.5D ConvNeXt reader" (Kaggle dataset) | Report-derived soft labels; gold held out | ConvNeXt-Tiny 2.5D MIL, 3 folds | n/a | 0.929 public alone; 0.944 in a 30% rank blend [A, search snippet] | [A] | Yes | A single good 2.5D reader on report labels reaches ≈0.93 public |
| mmatmath/rsna-knee-two-stage | Local **Qwen3-4B** (thinking off), strict 1/0/null; gold replaces weak labels | ConvNeXt-base (DINOv3 weights) 2.5D with target-specific attention MIL → small transformer over frozen embeddings | Round-robin folds by study; caveat: main CV is vs weak labels | None reported | [V, README] | Yes | Same local model family as here; "null" kept as missing |
| joshuaziel gpt-5.6-luna labels (2026-09-18) | DSPy "Consensus" program, verdict + confidence + evidence basis (STATED / DESCRIBED / NOT_MENTIONED) | n/a | Evaluated against gold | Not stated on the page | [V, page] | Yes as external data | States that gold labels came from an independent MRI read |
| skarin "58 Gold Labels vs 4 Teachers" (Kaggle notebook) | Compares 4 public label sets ("teachers"), class-wise selection | n/a | Cross-fitted on gold | Numbers not visible without login | [A, headings only] | Yes | Flags duplicate reports and split leakage, and that the protocol flags merge two properties |
| hengck23 (Kaggle discussion 747658) | Comment: labels are close to gold with systematic errors; proposes minimal corrections | n/a | n/a | n/a | [A] | Yes | One concrete label example given there **did not reproduce** on the local data |

## 5. Comparison of the strongest options for this dataset

| Option | Expected label quality vs gold (macro AUC) | Language coverage | Compute here | Calibration | Recommendation |
|---|---|---|---|---|---|
| Rules (ConText-style, 9 languages) | Moderate (measured in `label_validation_report.md`) | Only as good as the lexicon | Seconds | Ordinal states only | Keep as a transparent, independent ensemble member and evidence provider |
| Local LLM, forced-choice assertion probabilities (Qwen3-4B, full reports) | Measured in `label_validation_report.md` | Native multilingual | ≈2 h for 4,407 reports on a 6 GB GPU | Token probabilities, calibrated out-of-fold | Main local labeler |
| Public hosted-LLM label sets (gpt56sol, report_v2) | 0.835 / 0.870 measured here, but their prompts were probably tuned on gold, so this is optimistic | Good | none | Unknown | Use as ensemble members; never weight them by in-sample gold AUC |
| Ensemble (rank-average of independent sources) | Usually ≥ best single source when errors differ | Inherits | trivial | Re-calibrate out-of-fold | Recommended final teacher |
| Label model (Dawid-Skene / CROWDLAB) | Unreliable with 3–5 correlated sources and 58 gold | n/a | low | n/a | Not recommended |

## 6. Recommendation for this project

1. Treat report-derived labels as **noisy proxies of an image read**. Optimise them for downstream image-model AUC on gold-held-out validation, not for gold agreement alone.
2. Use **assertion categories** (present / possible / not mentioned / absent / prior) rather than a single probability. "Not mentioned" is not "absent" for findings that are routinely unreported (Synovitis, Fracture, Contusion, OA). Test masking vs. negative mapping downstream.
3. Use **full reports** with Unicode normalisation; the old 1,200-character truncation and the Greek micro-sign artefact both destroy information.
4. Build the teacher as a **rank-average ensemble of independent sources** (local LLM + rules + public sets). Fit any calibration or weights only out-of-fold on gold.
5. Above all, the competition gap is in the **image model**. Public evidence shows single 2.5D readers on report labels reach ≈0.92–0.93 public LB, against 0.775 for the current frozen-feature v2. Label improvements are second-order compared with fine-tuning a 2.5D backbone at higher resolution (see `FINAL_RESEARCH_AND_DATA_AUDIT.md`).

## References (links)

- Jain S. et al. VisualCheXbert (2021): <https://arxiv.org/abs/2102.11467>
- Jain S. et al. Effect of Radiology Report Labeler Quality on Deep Learning Models (2021): <https://arxiv.org/abs/2104.00793>
- Smit A. et al. CheXbert (EMNLP 2020): <https://aclanthology.org/2020.emnlp-main.117>
- Peng Y. et al. NegBio (AMIA 2018): <https://arxiv.org/abs/1712.05898>
- Irvin J. et al. CheXpert (AAAI 2019): <https://arxiv.org/abs/1901.07031>
- Mukherjee P. et al. Vicuna for labeling radiology reports, *Radiology* 2023: <https://pubs.rsna.org/doi/10.1148/radiol.231147>
- On-premise open-source LLMs for structuring reports (German/English), *Eur Radiol* 2024: <https://link.springer.com/article/10.1007/s00330-024-11074-y>
- Bien N. et al. MRNet, *PLOS Med* 2018: <https://doi.org/10.1371/journal.pmed.1002699>
- Zhao R. et al. fastMRI+, *Sci Data* 2022: <https://doi.org/10.1038/s41597-022-01255-z>
- Ilse M. et al. Attention-based deep MIL (ICML 2018): <https://arxiv.org/abs/1802.04712>
- Goh H.W., Tkachenko U., Mueller J. CROWDLAB (2022): <https://arxiv.org/abs/2210.06812>
- Ratner A. et al. Snorkel (VLDB 2017): <https://arxiv.org/abs/1711.10160>
- Bishnoi Y. HF blog: <https://huggingface.co/blog/bishnoiyash/rsna-competetion>
- aditbytes repo: <https://github.com/aditbytes/kaggle-rsna-knee-abnormality>
- mmatmath repo: <https://github.com/mmatmath/rsna-knee-two-stage>
- goodpjw2008 reader: <https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader>
- gpt-5.6-luna labels: <https://www.kaggle.com/datasets/joshuaziel/rsna-knee-mri-report-labels-gpt-5-6-luna>
- skarin notebook: <https://www.kaggle.com/code/skarin/rsna-knee-58-gold-labels-vs-4-teachers>
- Kaggle discussion 747658: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658>
