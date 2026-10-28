"""Build candidate A-strict+ (plan section 10c): goodpjw2008's 0.958 notebook (v27) with its R6 stage replaced by
heliosli's S6 stage from their 0.954 notebook. Everything else is copied unchanged.

fastMRI-free by construction: no R6, no R7. External data: OAI (nartaa's CoAtNets), SKM-TEA (`anatomy.pt`, S6's anchors).

Cells:
  new header + heliosli's credits/acknowledgement (verbatim)
  v27 Part A1 (nartaa 384), Part A1b (nartaa 224 + mirror)           unchanged
  heliosli S6 stage + controller                                      unchanged except one flag (see EDITS)
  v27 Part A3 blend                                                   S6 weights instead of R6 weights (see EDITS)
  v27 Part B reader at 1/10                                           unchanged

Every edit is an exact string replacement that must match once. The output is checked: cells parse as Python, and the
R6/R7 identifiers appear nowhere.

Usage: python -I src/emergency/build_a_strict_plus.py <v27.ipynb> <heliosli_restored.ipynb> <out_dir>
"""
import ast
import hashlib
import json
import os
import sys

V27_SHA_NOTE = "goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958 (version 27, pulled 2026-10-19)"
S6_SHA_NOTE = "heliosli/rsna-knee-blend-gold-dataset-restored (version 1, pulled 2026-10-19)"

HEADER = """# RSNA Knee: nartaa CoAtNet 384 + 224, heliosli S6, goodpjw2008 reader (fastMRI-free variant of the 0.958 recipe)

A copy of [goodpjw2008's 0.958 notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) (version 27) with one change: its compact model **R6 is replaced by heliosli's S6**, scored exactly as in [heliosli's 0.954 notebook](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-dataset-restored).

**Why:**
- R6's trainer (version 9.8) adds a fastMRI+ auxiliary task, and R6 was trained from a run named `cnx_r6fm`.
- fastMRI access requires an institutional affiliation, which the host's external-data test excludes.
- S6's trainer has no such option.

**What is unchanged:**

| Part | Source | Change |
|---|---|---|
| A1 | nartaa's 384-px CoAtNet | none (byte-identical in both source notebooks) |
| A1b | nartaa's 224-px checkpoint with the mirror view, 35 % of the Raptor slot (after varshithakolupuri) | none |
| A2 | heliosli's S6 stage | none, except that the controller no longer stops at 3 % unscored studies; the blend gives such a study Raptor's rank, as v27 does for R6 |
| A3 | v27's blend | Raptor 0.7 / S6 0.3 for every finding (heliosli's S6 weights) instead of R6's 0.5 with per-finding overrides |
| B | goodpjw2008's 2.5D reader at 1/10 | none |

**External data:**
- OAI, through nartaa's CoAtNet weights (acknowledgement below).
- SKM-TEA, through `anatomy.pt`, which places S6's knee-centred views.
- No fastMRI, fastMRI+, KneeCoT or other external images.
- Labels for all models come from the competition's own reports.

**Scores:**
- A blend of these parts has not been scored before.
- Measured by others: nartaa + S6, 0.954; the same with the 224 checkpoint, still 0.954 at three decimals, with a better rank (varshithakolupuri).
- Nothing in this notebook is tuned on the gold studies or on the leaderboard.

All credit for the models and code goes to nartaa, heliosli, goodpjw2008, varshithakolupuri and Dread Development (links below). Both source notebooks are released under Apache 2.0.
"""

A2_MD = """---
# Part A2 · heliosli's compact model S6 (unchanged from heliosli's 0.954 notebook)

Bundle `209a32c8b50a6b1c` from `heliosli/rsna-knee-compact-v1`: four ConvNeXt-small fold models trained on the competition's studies (gold excluded), with knee-centred views placed by `anatomy.pt` (SKM-TEA). Every file is checked against the bundle manifest. The stage writes `/kaggle/working/ours_sub.csv`, the same interface as v27's R6 stage.
"""

A3_MD = """---
# Part A3 · the rank blend, 224 in the Raptor slot

Raptor rank = 0.65 x rank(384 model) + 0.35 x rank(224 model); compact rank = rank(S6); final = 0.7 x Raptor + 0.3 x S6 for every finding (heliosli's S6 weights, fixed before any run). This writes the submission that Part B nudges.
"""

# (cell label, old, new): each old string must occur exactly once in that cell
EDITS = {
    "s6_controller": [(
        "'--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS)], env=env).returncode",
        "'--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS),\n"
        "                     '--max-fallback-fraction', '1.0'], env=env).returncode   # [a-strict+] as v27 runs R6: the blend "
        "below gives a study S6 did not score Raptor's rank (more than 50 % unscored stops the notebook)",
    )],
    "blend": [
        ("# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5.",
         "# Stage 3: rank blend with weights fixed before any run: Raptor 0.7, S6 0.3 for every finding ([a-strict+] heliosli's S6 weights)."),
        ("W_RAPTOR, W_OURS = 0.5, 0.5\n",
         "W_RAPTOR, W_OURS = 0.7, 0.3   # [a-strict+] heliosli's S6 weights (their 0.954 notebook); v27 used 0.5 / 0.5 for R6\n"),
        ("W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}",
         "W_RAPTOR_C = {}   # [a-strict+] the per-finding overrides were set for R6; S6 keeps one weight for every finding\n"
         "_V27_R6_OVERRIDES = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # unused, for reference"),
    ],
}

FORBIDDEN = ["fe055406", "r7cnxfsss", "rsna-knee-r6", "8c7b666e", "6508ce69"]


def src(nb, i):
    return "".join(nb["cells"][i]["source"])


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def apply(text, label):
    for old, new in EDITS.get(label, []):
        n = text.count(old)
        assert n == 1, f"{label}: expected one match, found {n}: {old[:80]!r}"
        text = text.replace(old, new)
    return text


def main():
    v27_path, hel_path, out_dir = sys.argv[1:4]
    v = json.load(open(v27_path, encoding="utf-8"))
    h = json.load(open(hel_path, encoding="utf-8"))
    assert len(v["cells"]) == 19 and len(h["cells"]) == 7, "unexpected source notebook layout"
    assert "BUNDLE_ID = 'fe055406fad6a12f'" in src(v, 9) and "deadline = T0" in src(v, 10)
    assert "BUNDLE_ID = '209a32c8b50a6b1c'" in src(h, 4) and "deadline = T0" in src(h, 5)
    same = lambda a, b: a.rstrip("\n") == b.rstrip("\n")   # the copies differ only by a trailing newline
    assert same(src(v, 3), src(h, 2)) and same(src(v, 4), src(h, 3)), "the 384 stage differs between the sources"
    assert "W_RAPTOR, W_OURS = 0.5, 0.5" in src(v, 12)

    cells = [
        cell("markdown", HEADER),
        cell("markdown", src(h, 1)),                    # credits + required OAI acknowledgement, verbatim
    ]
    cells += [cell(v["cells"][i]["cell_type"], src(v, i)) for i in range(2, 8)]   # A1, A1b: unchanged
    cells += [cell("markdown", A2_MD), cell("code", src(h, 4)), cell("code", apply(src(h, 5), "s6_controller"))]
    cells += [cell("markdown", A3_MD), cell("code", apply(src(v, 12), "blend"))]
    cells += [cell(v["cells"][i]["cell_type"], src(v, i)) for i in range(13, 19)]  # Part B: unchanged

    nb = {"cells": cells, "metadata": v["metadata"], "nbformat": 4, "nbformat_minor": v.get("nbformat_minor", 4)}
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    for f in FORBIDDEN:
        assert f not in text, f"forbidden identifier in the output: {f}"
    for i, c in enumerate(cells):                       # every code cell parses (cell magics written to files are skipped)
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if not s.startswith("%%writefile"):
                ast.parse(s)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "rsna-knee-a-strict-plus.ipynb")
    open(out, "w", encoding="utf-8").write(text)
    meta = {
        "id": "nishkarshsinghal/rsna-knee-a-strict-plus",
        "title": "rsna-knee-a-strict-plus",
        "code_file": "rsna-knee-a-strict-plus.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "goodpjw2008/rsna-knee-2-5d-convnext-reader",
            "heliosli/rsna-knee-compact-v1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
        # heliosli's image (their 0.954 notebook): the two stages that stop the notebook on failure (nartaa 384 and S6) both
        # ran on it. The 224 stage and Part B ran on v27's image (37c64f7d...); both fall back to the A3 blend if they fail.
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4",
    }
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("wrote", out, len(cells), "cells", hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
    print("sources:", V27_SHA_NOTE, "|", S6_SHA_NOTE)


if __name__ == "__main__":
    main()
