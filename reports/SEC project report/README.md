# B.Tech project report: Knee Abnormality Detection Using Deep Learning

Final deliverable: `Knee_Abnormality_Detection_BTech_Report.pdf`.

## Folder contents

| Path | What it is |
|---|---|
| `Knee_Abnormality_Detection_BTech_Report.tex` | main LaTeX file (preamble with the FOT ECE format settings) |
| `chapters/` | title page, preliminary pages, Chapters 1–4, references, appendix |
| `figures/` | every figure used in the report (PDF charts, workstation screenshots, university logo) |
| `data/` | aggregate tables behind the charts (counts, rates, AUCs, confidence intervals); no per-study data, report text or images |
| `scripts/` | scripts that produced `data/` and `figures/` |
| `B.Tech Project Report Format FOT_ECE Department-1.pdf` | the department's format guide that the report follows |

## Compiling

The report needs only the files in this folder. With MiKTeX or TeX Live:

```
pdflatex Knee_Abnormality_Detection_BTech_Report.tex
pdflatex Knee_Abnormality_Detection_BTech_Report.tex
pdflatex Knee_Abnormality_Detection_BTech_Report.tex
```

Packages used: newtx (Times), geometry, setspace, titlesec, tocloft, caption, booktabs, tabularx, longtable, makecell, tikz, xurl, hyperref, babel (British).

## Regenerating the figures (optional; needs the project's run outputs on this machine)

```
python scripts/collect_results.py          # run outputs -> data/*.csv (aggregates only)
python scripts/prepare_submissions.py <kaggle submissions export.csv>
python scripts/make_figures.py             # data/*.csv -> figures/*.pdf
node scripts/capture_workstation.mjs <out> <chrome-profile>   # workstation screenshots (server on 127.0.0.1:8000)
python scripts/prepare_screenshots.py <out>   # cuts figures/ws_*.png (add --mask to hide the MRI canvases)
python scripts/order_references.py         # after a compile: references in order of first citation
```

## Items the team should confirm before printing

- The academic session on the title page (written as 2026–2027).
- The project period in the declaration (written as August 2026 to November 2026, from the first code commit and first Kaggle submission to the present).
