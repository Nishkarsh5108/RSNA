# Competition rules audit: RSNA Knee Abnormality Detection (Kaggle, 2026)

Audit date: 2026-10-17. Kaggle pages are JavaScript-rendered and there are no Kaggle API credentials on this machine, so the official Overview, Data and Rules tabs were read through a public reader proxy (r.jina.ai). Where a rule could not be read verbatim, I took the conservative reading.

## 1. Facts verified from official sources

| Item | Finding | Source |
|---|---|---|
| Task | Study-level prediction of 12 binary findings from knee MRI | Kaggle Overview and Data tabs |
| Targets (exact column names) | `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture` | Overview submission header; local `sample_submission.csv` |
| Official target definitions | ACL = anterior cruciate ligament **injury**; MCL = medial collateral ligament **injury**; Medial/Lateral Meniscus = meniscus **tear**; Medial/Lateral OA = OA of the medial/lateral **tibiofemoral** compartment; PF OA = patellofemoral OA; Effusion = joint effusion (excess fluid); Synovitis = inflammation of the joint lining; Baker's = Baker's cyst; Contusion = bone contusion (bone bruise); Fracture = fracture. All 0/1. | Kaggle Data tab |
| Metric | Macro-averaged ROC-AUC: Score = (1/12) Σ AUC_i | Overview, Evaluation section |
| Submission | `submission.csv`, one row per test `StudyInstanceUID`, a probability per target | Overview; `sample_submission.csv` |
| Labeled training data | Only "a small subset" of training studies have per-condition labels. Locally verified: **58 of 4,407**. | Data tab; `train.csv` |
| Reports | Present for all training studies, in several languages. **Not provided for test studies.** | Data tab |
| Test set | About 1,300 studies, swapped in at scoring time (the local `test.csv` holds 3 example studies) | Data tab |
| Prevalence warning | Prevalence is "not guaranteed to match across the training, public leaderboard, and final evaluation sets" | Data tab |
| DICOM | Each file stripped to an allowlist of 86 tags (locally: 63 elements in the sampled header); mixed transfer syntaxes (Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE) in the official data | Data tab. The **local download is 100% Explicit VR LE**, so the compressed syntaxes are a hidden-test concern (see `EDA_FINAL.md` R12). |
| Code competition | Notebook submissions only; CPU and GPU runtime ≤ 9 h; **internet disabled** | Overview, Code Requirements |
| External data and models | "Freely and publicly available external data is allowed, including pre-trained models" | Overview, Code Requirements |
| Timeline | Final submission deadline **2026-10-22 23:59 UTC** (5 days after this audit); entry/merger deadline 2026-10-15 (passed) | Overview, Timeline |
| Prizes | $77k total, including an Efficiency Track that combines AUC and runtime | Overview, Prizes |
| Team / submissions | Max team size 5; 5 submissions per day; up to 2 final submissions | Rules, Section 2 |
| Winner licence | CC-BY-NC 4.0 for winning code | Rules |
| Data licence | MIRA licence. Competition data may be used "for any purpose, commercial or non-commercial" within the rules. | Rules |
| Data security | Participants agree not to "transmit, duplicate, publish, redistribute or otherwise provide" competition data to anyone not participating | Rules (competition-specific section) |
| Hand labeling | General Rule 3.4(b) forbids using "information from hand labeling or human prediction of the validation dataset or test data records" | Rules |

## 2. Rules as they apply to the label-generation workflow

| Question | Answer | Rule / evidence | Practical implication |
|---|---|---|---|
| May the training reports be used to derive labels? | **Yes.** The Data tab says participants may derive labels for unlabeled studies from the reports. | Data tab | Report-derived pseudo-labels are the intended route |
| May pre-trained open-weight LLMs be used? | **Yes**, if freely and publicly available. Qwen3-4B-Instruct-2507 (Apache-2.0, public on Hugging Face) qualifies. | Overview, Code Requirements | The local LLM labeler used here is compliant |
| May public third-party label sets (Kaggle Datasets of LLM labels) be used? | **Yes, as external data,** provided they are public and equally accessible. Their own compliance is the author's responsibility, and their provenance must be documented. | External data rule | `gpt56sol`, `report_v2` etc. may be used, but their provenance is not recorded in this repository (see `dataset_inventory.md`) |
| May reports be sent to a hosted LLM API (Gemini, OpenAI, Anthropic, …)? | **Ambiguous; treated as NOT allowed here.** The data-security clause forbids providing data to parties not in the competition, and an API provider is such a party. The Rules separately say a small LLM subscription is "reasonable" in cost, but that concerns cost, not data transfer. | Data-security clause | **No report text left this machine during this work.** Note: `data/main_data/without_labels.csv` and `docs/multilingual_llm_medical_classification_chat.md` show that the 58 gold reports were previously processed with a hosted model (Gemini Flash). That is a compliance grey zone, and I recommend confirming with the host before any similar use. |
| Are teacher models, pseudo-labeling and knowledge distillation allowed? | **Yes.** No rule prohibits them. | Absence of a prohibition; external-data rule | Allowed |
| Can reports be used at inference? | **No, they are not available.** Test rows have no `Report`. | Data tab | Text can only supervise training. **Any model input derived from reports is leakage and will fail at test time.** |
| Hand labeling of training reports or images? | Not forbidden for **training** records. Forbidden for validation/test records. | Rule 3.4(b) | Manual reading of training reports for rule development is allowed. Never hand-label test images. |
| Network access during submission? | **No.** | Code Requirements | Model weights (e.g. backbones) must be attached as Kaggle Datasets. The label generator does not need to run at submission time. |
| Runtime limits | ≤ 9 h per submission notebook. About 1,300 test studies × about 186 slices per study. | Overview; inventory | DICOM decode plus a 2.5D model must fit in 9 h on Kaggle T4/P100 |

## 3. Ambiguities and the conservative choices made

1. **Hosted LLM APIs for report text.** Treated as not allowed, so only a local model was used. The previous hosted-LLM use on the 58 gold reports should be disclosed or confirmed with the host. The rules do not explicitly cover it.
2. **Origin of the 58 gold labels.** There is no official statement on the Kaggle pages read here. Several independent participants state the gold labels came from an independent **image** read (radiologists who did not use the report): the gpt-5.6-luna dataset description, a Kaggle discussion comment by hengck23, and the Hugging Face write-up by Y. Bishnoi. RSNA's announcement says the evaluation data was "annotated by expert radiologists". Here this is treated as **probable but unofficial**. One concrete third-party example of a report/gold contradiction (study `…12606657226568558340797193167488111973`, "Effusion = 1 but report says no effusion") **did not reproduce** on the local `train.csv`, where that study has Effusion = 0.
3. **Public label datasets created with hosted LLMs.** If a public dataset's author sent reports to a hosted API, any compliance question rests with that author. Using the published dataset is covered by the external-data rule, but I recommend a note in any writeup.
4. **Reasonableness standard for compute.** Running a 4B model locally is clearly within it.

## 4. Compliance risks found in this repository (action needed by the owner)

1. **The GitHub remote is public and contains competition data.** `origin = https://github.com/Nishkarsh5108/RSNA` showed a "Public" label on 2026-10-17. Git tracks the following (commit `b903527`):
   - `data/main_data/train (1).csv`, byte-identical to the official `train.csv`, including **all 4,407 radiology reports and the 58 gold labels**;
   - `train_series.csv`, test CSVs, and several label files that embed the `Report` column;
   - a raw competition DICOM at the repo root (`1.2.826.0.1.3680043.8.498.11421887690184506354786744807184755984.dcm`).

   This very likely violates the data-security / no-redistribution rule and could lead to disqualification. **Recommendation:** make the repository private now. To fully remove the data, rewrite history (e.g. `git filter-repo --path data/main_data --path <dcm> --invert-paths`) and force-push. I did **not** change the remote, the history, or any tracked file. That decision is the owner's.
2. **Hosted-LLM processing of gold reports** (Gemini, see §3.1). Disclose this or confirm it with the host.
3. **New outputs are git-ignored.** `data/labels_v3/` and `reports/eda/` contain report excerpts (evidence spans), per-study labels or image-derived figures. I added them to `.gitignore` so they cannot be pushed by accident. The top-level `reports/*.md` files contain only aggregate statistics and study IDs, with no report text.

## 5. Compliance of the workflow delivered here

| Component | Compliant? | Note |
|---|---|---|
| `src/labeling/llm_assertion_labeler.py` (Qwen3-4B-Instruct-2507, local) | Yes | Public open-weight model; reports never leave the machine |
| `src/labeling/rule_extractor.py` | Yes | Hand-written rules developed on non-gold training reports |
| Use of existing public label sets as ensemble members | Yes, as external data | Provenance must be documented (currently missing in the repo) |
| Downstream experiments | Yes | Training data only; gold used only for evaluation |
| Kaggle inference | Not part of this work | Must not take any report-derived input at test time |

## Sources

- Kaggle competition pages (Overview / Data / Rules): <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>
- RSNA announcement (2026-08-05): <https://www.rsna.org/news/2026/august/ai-challenge-knee-mri>
- RSNA challenge page: <https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge>
- gpt-5.6-luna label dataset (statement on how the gold labels were made): <https://www.kaggle.com/datasets/joshuaziel/rsna-knee-mri-report-labels-gpt-5-6-luna>
- Kaggle discussion "RSNA Knee MRI reports — LLM label-extraction benchmark": <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747658>
- Hugging Face write-up (Y. Bishnoi, 2026-08-09): <https://huggingface.co/blog/bishnoiyash/rsna-competetion>
