"""Turn the Kaggle CLI export of our submission history into ../data/submissions_owner.csv (no dates).

The export is produced with the project's guarded, read-only wrapper:
    python kg.py competitions submissions rsna-knee-abnormality-detection --csv > submissions_export.csv
    python prepare_submissions.py submissions_export.csv
Each submission keeps its Kaggle reference number, its order, the scores and a short description; the stage
labels follow the development stages described in Chapter 3 of the report.
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
STAGE = {  # submission ref -> (stage, short description)
    55906095: ("Stage 1", "v1: frozen EfficientNet-B0 on a third-party cache"),
    55966953: ("Stage 1", "v2: frozen EfficientNet-B0 on our own DICOM pipeline"),
    57336780: ("Stage 3", "2.5D MIL ConvNeXt-T, 3 seeds (F1-F3), shift TTA"),
    57365203: ("Stage 3", "F1-F3 + high-resolution model E9"),
    57384567: ("Stage 4", "A-strict: public CoAtNet (OAI) + public S6"),
    57384570: ("Stage 4", "A-strict+: public 0.958 notebook with R6 replaced by S6"),
    57406523: ("Stage 4", "final-strict: A-strict + public s6fall (fastMRI-free)"),
    57444580: ("Stage 4", "final-max v1: public CoAtNet + R6 notebook + s6fall"),
    57460890: ("Stage 4", "final-max v2: s6fall scored inside the R6 pass"),
    57480525: ("Stage 4", "final-max v3: + our fine-tuned s6ft_v2, s6ft_v1"),
    57480912: ("Stage 4", "final-strict-v2: + our fine-tunes"),
    57512015: ("Stage 4 (last day)", "final-max v4: specialist routing (no score returned)"),
    57519582: ("Stage 4 (last day)", "final-strict-v3: specialist routing fitted on gold-58"),
    57521956: ("Stage 4 (last day)", "final-max (kernel v8): stopped by its 8.3 h deadline"),
    57531417: ("Stage 4 (last day)", "final-max (kernel v9): same blend, later version"),
    57534028: ("Stage 4 (last day)", "final-max (kernel v10): stride-2 master blend"),
    58350230: ("Stage 5", "box-pooling, fused labels, fold 0 (A2)"),
    58350257: ("Stage 5", "box-pooling, fused labels, folds 0+1 (A2 + C1)"),
    58369148: ("Stage 5", "A1 with per-finding attention sharpening"),
    58369192: ("Stage 5", "mean of A0, A1, A2"),
    58369222: ("Stage 5", "A1 with attention sharpening s = 2"),
    58369269: ("Stage 5", "mean of A0, A1, A2, C1"),
}

src = sys.argv[1]
lines = [l for l in open(src, encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("Warning")]
t = pd.read_csv(pd.io.common.StringIO("\n".join(lines)))
t = t.sort_values("date").reset_index(drop=True)
t["order"] = range(1, len(t) + 1)
t["stage"] = t["ref"].map(lambda r: STAGE[r][0])
t["what"] = t["ref"].map(lambda r: STAGE[r][1])
out = t[["order", "ref", "stage", "what", "publicScore", "privateScore"]].rename(columns={"publicScore": "public", "privateScore": "private"})
out.to_csv(OUT / "submissions_owner.csv", index=False)
print(out.to_string())
