import { useEffect, useRef, useState } from "react";
import { FINDINGS, PROJECT } from "../lib/findings";

// ---------------------------------------------------------------- 01 findings index
export function Findings() {
  const max = Math.max(...FINDINGS.map((f) => f.prevalence));
  return (
    <section className="findings" id="findings">
      <header className="sec-head grid">
        <span className="label sec-idx">01 — Findings</span>
        <h2 className="h2 sec-title">Twelve findings, one probability each</h2>
        <p className="lede sec-lede">
          The model reads a whole study and outputs twelve independent probabilities. The bars show how often each
          finding is present among the 4,349 labelled training studies, so a score can be read against its base rate.
        </p>
      </header>
      <div className="grid">
        <div className="findings-table" role="table" aria-label="The twelve findings">
          <div className="f-row f-row--head" role="row">
            <span className="label" role="columnheader">#</span>
            <span className="label f-code" role="columnheader">Code</span>
            <span className="label" role="columnheader">Finding</span>
            <span className="label f-region" role="columnheader">Where the model looks</span>
            <span className="label f-prev-h" role="columnheader">Prevalence</span>
          </div>
          {FINDINGS.map((f, i) => (
            <div className="f-row" role="row" key={f.key}>
              <span className="num f-idx">{String(i + 1).padStart(2, "0")}</span>
              <span className="label f-code">{f.short}</span>
              <span className="f-name">{f.name}</span>
              <span className="body-s f-region">{f.region}</span>
              <span className="f-prev">
                <span className="f-bar" style={{ ["--w" as string]: `${(f.prevalence / max) * 100}%` }} />
                <span className="num">{(f.prevalence * 100).toFixed(1)}%</span>
              </span>
            </div>
          ))}
          <p className="label f-note">
            Prevalence: share of training studies whose label (mean of the 1st- and 7th-place image-read label sets) is ≥ 0.5.
          </p>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- 03 contributions
const FOUNDATIONS = [
  {
    t: "Three network architectures and their training scripts",
    d: "Side classifier, ROI detector and box-pooling classifier: the public 1st-place solution by tascj0 (Team Kangaroo). The detector and side-classifier weights used here are theirs.",
    m: "public notebooks",
  },
  {
    t: "Image-read label sets",
    d: "Per-study labels made by reading the images as well as the reports, published by tascj0 (1st place) and songqizhou (7th place).",
    m: "CC BY 4.0 · CC BY-NC-SA 4.0",
  },
  {
    t: "Data",
    d: "RSNA 2026 knee abnormality detection competition: 4,407 training studies, 58 with image-read reference labels.",
    m: "competition licence",
  },
];

const OURS = [
  {
    t: "Showed that labels were the bottleneck",
    d: "A controlled swap: our own model retrained with only the label file changed. Report-only labels → image-read labels.",
    m: "+0.036 gold-58 AUC · 95% CI +0.017 to +0.057",
  },
  {
    t: "Fused the two public label sets",
    d: "Trained on the mean of the 1st- and 7th-place labels. In the box-pooling model the source barely matters, so fusion is a safe compromise, not a measured gain.",
    m: "gold-58: 0.938 fused vs 0.939 / 0.939",
  },
  {
    t: "Rebuilt the inference pipeline",
    d: "The public inference depended on TensorRT engines and a C++ reader. We rebuilt it from the three training scripts and verified it.",
    m: "corr 0.996 vs public notebook · 0.954 published → 0.955 reproduced",
  },
  {
    t: "Trained the models",
    d: "One ConvNeXt-T per run, 8 epochs on 2 × T4, about 6–7 h each: fold 0 under each label set, then further folds on the fused labels; validation predictions kept per series.",
    m: "MODELS",
  },
  {
    t: "Kept the evaluation honest",
    d: "Final ensemble pre-registered before its score existed. Models compared on a held-out fold scored against the label source they were not trained on. Ideas inside the ≈ 0.007 seed noise were rejected.",
    m: "sharpening +0.0005 · series weights +0.0007 · own model −0.001",
  },
  {
    t: "Built this workstation",
    d: "Local GPU inference with exact per-cell attribution, checked against the model's own logits on every run.",
    m: "≈ 1.2 GB peak VRAM per study · 127.0.0.1",
  },
];

export function Contributions({ models }: { models: number | null }) {
  return (
    <section className="contrib" id="contributions">
      <header className="sec-head grid">
        <span className="label sec-idx">03 — Contributions</span>
        <h2 className="h2 sec-title">What we built, and what we built on</h2>
        <p className="lede sec-lede">
          The architecture is not ours. Our work is the label study, the fusion, the verified pipeline, the training runs,
          the evaluation discipline and this local system.
        </p>
      </header>
      <div className="grid contrib-cols">
        <div className="contrib-col contrib-col--base">
          <h3 className="label contrib-h">Built on</h3>
          <ol className="ledger">
            {FOUNDATIONS.map((f, i) => (
              <li key={f.t}>
                <span className="num ledger-idx">{String.fromCharCode(65 + i)}</span>
                <div>
                  <p className="ledger-t">{f.t}</p>
                  <p className="body-s">{f.d}</p>
                  <p className="label ledger-m">{f.m}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="contrib-col contrib-col--ours">
          <h3 className="label label--fg contrib-h">
            <span className="dot dot--accent" /> Ours
          </h3>
          <ol className="ledger">
            {OURS.map((f, i) => (
              <li key={f.t}>
                <span className="num ledger-idx">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <p className="ledger-t">{f.t}</p>
                  <p className="body-s">{f.d}</p>
                  <p className="label ledger-m ledger-m--accent">
                    {f.m === "MODELS" ? (models ? `${models} checkpoints loaded in the workstation now` : "checkpoints read from the local server") : f.m}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- 04 results
const RUNS = [
  { id: "C", name: "Competition best (our own model)", priv: 0.943, pub: 0.958, note: "rank 490 · during the competition" },
  { id: "A0", name: "1st-place labels · fold 0", priv: 0.955, pub: 0.963, note: "published recipe reproduced (0.954)" },
  { id: "A1", name: "7th-place labels · fold 0", priv: 0.957, pub: 0.967, note: "label swap" },
  { id: "A0+A1", name: "A0 + A1", priv: 0.958, pub: 0.967, note: "two-model mean" },
  { id: "A2", name: "Fused labels · fold 0", priv: 0.958, pub: 0.968, note: "mean of both label sets", hi: true },
  { id: "A2+C1", name: "Fused labels · folds 0 + 1", priv: 0.958, pub: 0.968, note: "two-fold mean" },
];

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function Results() {
  const [ref, width] = useWidth<HTMLDivElement>();
  const narrow = width < 640;
  const lo = 0.94, hi = 0.97;
  const left = narrow ? 12 : 250, right = 24;
  const rowH = narrow ? 58 : 46;
  const top = 46;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (width - left - right);
  const height = top + RUNS.length * rowH + 40;
  const ticks = [0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97];
  return (
    <section className="results" id="results">
      <header className="sec-head grid">
        <span className="label sec-idx">04 — Results</span>
        <h2 className="h2 sec-title">0.943 → 0.958</h2>
        <p className="lede sec-lede">
          Macro ROC-AUC over the twelve findings on the competition's hidden test set (about 1,300 expert-labelled
          studies). Post-competition runs are late submissions: they are scored, but cannot change the final ranking.
        </p>
      </header>
      <div className="grid">
        <div className="results-chart" ref={ref}>
          <svg width={width} height={height} role="img" aria-label="Private leaderboard scores by run">
            <rect x={x(0.957)} y={top - 22} width={x(0.959) - x(0.957)} height={height - top - 10} className="band" />
            <text x={x(0.958)} y={top - 28} className="fig-tag fig-dim" textAnchor="middle">RANKS 3–9</text>
            <line x1={x(0.96)} x2={x(0.96)} y1={top - 22} y2={height - 32} className="winner" />
            <text x={x(0.96) + 6} y={top - 10} className="fig-tag">1ST · 0.960</text>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={height - 30} y2={height - 24} className="fig-rule" />
                <text x={x(t)} y={height - 8} className="fig-tag fig-dim" textAnchor="middle">
                  {t.toFixed(3)}
                </text>
              </g>
            ))}
            <line x1={left} x2={width - right} y1={height - 30} y2={height - 30} className="fig-rule" />
            {RUNS.map((r, i) => {
              const y = top + i * rowH + rowH / 2;
              return (
                <g key={r.id} className={r.hi ? "run run--hi" : "run"}>
                  <line x1={left} x2={width - right} y1={y} y2={y} className="run-rule" />
                  {narrow ? (
                    <text x={left} y={y - 12} className="fig-tag">
                      {r.id} · {r.name}
                    </text>
                  ) : (
                    <>
                      <text x={0} y={y - 3} className="run-id">{r.id}</text>
                      <text x={0} y={y + 13} className="fig-tag fig-dim">{r.name}</text>
                    </>
                  )}
                  <line x1={x(r.priv)} x2={x(r.pub)} y1={y} y2={y} className="run-span" />
                  <circle cx={x(r.pub)} cy={y} r={3.5} className="run-pub" />
                  <circle cx={x(r.priv)} cy={y} r={5} className="run-priv" />
                  <text x={x(r.priv) - 10} y={y + 4} className="fig-tag run-val" textAnchor="end">
                    {r.priv.toFixed(3)}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="results-legend">
            <span className="label"><i className="lg lg--priv" /> private leaderboard</span>
            <span className="label"><i className="lg lg--pub" /> public leaderboard</span>
            <span className="label"><i className="lg lg--hi" /> fused labels (the label set the later fold models use)</span>
          </div>
          <p className="body-s results-note">
            Differences of 0.002 or less are inside the test set's own noise, so 0.955–0.958 should be read as one
            level. A single model of this architecture, trained by us, reaches the scores of teams ranked 3–9; the
            1st-place ensemble scored 0.960. The four-model ensemble loaded in the workstation has not been scored on
            the leaderboard.
          </p>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- footer
export function Footer() {
  return (
    <footer className="footer">
      <div className="grid footer-cta">
        <p className="display footer-big">Read a study.</p>
        <div className="footer-cta-side">
          <p className="body-s">
            Drop a knee MRI study (a DICOM folder or a .zip) into the workstation. Files are parsed and run on this
            machine; nothing is uploaded anywhere else.
          </p>
          <a className="btn btn--accent" href="#/workstation">
            Open the workstation <span className="arrow">→</span>
          </a>
        </div>
      </div>
      <div className="hair" />
      <div className="grid footer-cols">
        <div className="footer-col">
          <span className="label">Notice</span>
          <p className="body-s">
            Research prototype. Not a medical device and not for diagnosis or treatment decisions. Probabilities are
            model outputs trained on machine-generated labels, not clinical findings.
          </p>
        </div>
        <div className="footer-col">
          <span className="label">Team</span>
          <p className="body-s">
            Nishkarsh Singhal · Anuj Chaubey · Vardhan Harsh Nigam
            <br />
            Supervision: Dr. Varun Saxena, Dr. Sangeeta Yadav
            <br />
            {PROJECT.institution}
          </p>
        </div>
        <div className="footer-col">
          <span className="label">Credits</span>
          <p className="body-s">
            Architectures, training scripts, detector and side-classifier weights: tascj0 (Team Kangaroo, 1st place).
            Labels: tascj0 (CC BY 4.0), songqizhou (7th place, CC BY-NC-SA 4.0). Data: RSNA 2026 knee abnormality
            detection challenge.
          </p>
        </div>
        <div className="footer-col">
          <span className="label">Runtime</span>
          <p className="body-s">
            All processing on 127.0.0.1. Fonts and scripts are served locally; the site makes no external requests.
          </p>
        </div>
      </div>
      <div className="grid footer-base">
        <span className="label">{PROJECT.name}</span>
        <span className="label">2026</span>
      </div>
    </footer>
  );
}
