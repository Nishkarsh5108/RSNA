import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { FINDINGS } from "../lib/findings";
import MethodFigure from "./MethodFigure";

gsap.registerPlugin(ScrollTrigger);

const STEPS: { title: string; body: string; facts: string[]; fig: string }[] = [
  {
    title: "A study is a set of series",
    body: "An MRI study holds several series: sagittal, coronal and axial planes, in proton-density, T1 or T2 weighting, with or without fat suppression. Each series is a stack of 2D slices, sorted along its slice normal before anything else happens.",
    facts: ["DICOM geometry: n = row × column direction", "4–9 series per study, 13–160 slices each"],
    fig: "Three series of one study, as stacks of slices",
  },
  {
    title: "Which knee is it?",
    body: "A small network reads every coronal and axial series and gives the probability that it shows a right knee. The study's mean decides. Left knees are mirrored so every knee is read as a right knee, and medial stays medial.",
    facts: ["ResNet-18d · 224 px · ≤ 64 slices", "mean p(right) ≥ 0.5 → right"],
    fig: "Side classification and mirroring",
  },
  {
    title: "Find the anatomy first",
    body: "A detector draws one 3D box per finding in every series: where the ACL, each meniscus, each compartment, the patellofemoral joint, the popliteal fossa should be. For each box edge it predicts a distribution along the axis; the edge is its expectation. A mirrored second pass is averaged in.",
    facts: ["ConvNeXt-T per slice · 160 × 160 · native depth ≤ 192", "3D top-down fusion · FiLM on plane, side, fat-sat, field of view"],
    fig: "Twelve boxes in one series; boundary distributions of the selected box",
  },
  {
    title: "Crop, then encode every slice",
    body: "Every slice is resampled to a 140 mm field. A 112 mm window around the knee's joint boxes is cut out at 256 px, and a ConvNeXt-T turns each slice into an 8 × 8 grid of feature cells, 768 numbers per cell.",
    facts: ["frame 140 mm @ 384 px → crop 112 mm @ 256 px", "cell ≈ 14 mm · ≤ 64 slices per series"],
    fig: "The crop rule and the feature grid",
  },
  {
    title: "Each finding reads only its box",
    body: "For each finding, the cells inside its box are averaged, weighted by how much of the cell lies in the box and by a learned in-box attention. One linear row per finding turns that average into a logit. During training only, two extra heads predicted lesion boxes on slices and the radiology report's text embedding.",
    facts: ["weight = box share × exp(attention), normalized", "logit = h_k · f̄ + b_k"],
    fig: "Box pooling for the selected finding",
  },
  {
    title: "Average series, then models",
    body: "A study's probability for a finding is the mean over its series, then the mean over the trained models. The models are fold checkpoints trained on the same recipe; the equal-weight ensemble was fixed before any test score was seen.",
    facts: ["study p = mean over series, then over models", "checkpoints read from the local server"],
    fig: "Series × models for the selected finding",
  },
  {
    title: "Why it said so",
    body: "Because the head is linear over a weighted mean, the logit splits exactly into one term per feature cell: its pooling share times the head's response to that cell. This is class activation mapping restricted to the box. The workstation draws these terms on the MRI and checks that they add up to the logit.",
    facts: ["exact decomposition, no gradients or sampling", "sum checked against the model's own output"],
    fig: "Per-cell contributions to one logit",
  },
];

export default function Method({ models }: { models: number }) {
  const root = useRef<HTMLDivElement>(null);
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [finding, setFinding] = useState("ACL");
  const touched = useRef(false);

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    const items = Array.from(el.querySelectorAll<HTMLElement>(".step"));
    let raf = 0;
    // on narrow screens the figure sits on top of the text, so a step becomes active lower in the viewport
    const line = window.innerWidth < 900 ? "78%" : "55%";
    const triggers = items.map((item, i) =>
      ScrollTrigger.create({
        trigger: item,
        start: `top ${line}`,
        end: `bottom ${line}`,
        onToggle: (self) => self.isActive && setStep(i),
        onUpdate: (self) => {
          if (!self.isActive) return;
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(() => setProgress(Math.round(self.progress * 40) / 40));
        },
      }),
    );
    return () => {
      cancelAnimationFrame(raf);
      triggers.forEach((t) => t.kill());
    };
  }, []);

  // While the box step is on screen and nobody has picked a finding, walk through them.
  useEffect(() => {
    if (step !== 2 || touched.current) return;
    const id = window.setInterval(() => {
      setFinding((f) => FINDINGS[(FINDINGS.findIndex((x) => x.key === f) + 1) % FINDINGS.length].key);
    }, 1800);
    return () => clearInterval(id);
  }, [step]);

  const pick = (key: string) => {
    touched.current = true;
    setFinding(key);
  };
  const chips = step === 2 || step >= 4; // the figures that depend on the chosen finding

  return (
    <section className="method" id="method" ref={root}>
      <header className="sec-head grid">
        <span className="label sec-idx">02 — Method</span>
        <h2 className="h2 sec-title">How a study is read</h2>
        <p className="lede sec-lede">
          Three networks run in sequence. The first finds the knee's side, the second finds the anatomy, the third reads
          each finding inside its own region. Scroll through the seven steps.
        </p>
      </header>
      <div className="method-body grid">
        <div className="method-figure">
          <div className="method-figure-inner crop">
            <div className="fig-head">
              <span className="label">FIG. {String(step + 1).padStart(2, "0")}</span>
              <span className="label label--fg">{STEPS[step].fig}</span>
              <span className="label fig-steps" aria-hidden="true">
                {STEPS.map((_, i) => (
                  <i key={i} className={i === step ? "on" : i < step ? "past" : ""} />
                ))}
              </span>
            </div>
            <MethodFigure step={step} progress={progress} finding={finding} models={models} />
            <div className={chips ? "chips chips--on" : "chips"} role="group" aria-label="Choose a finding">
              {FINDINGS.map((f) => (
                <button
                  key={f.key}
                  className={f.key === finding ? "chip chip--on" : "chip"}
                  onMouseEnter={() => chips && pick(f.key)}
                  onFocus={() => chips && pick(f.key)}
                  onClick={() => pick(f.key)}
                  tabIndex={chips ? 0 : -1}
                  title={f.region}
                >
                  {f.short}
                </button>
              ))}
            </div>
          </div>
        </div>
        <ol className="method-steps">
          {STEPS.map((s, i) => (
            <li key={i} className={i === step ? "step step--on" : "step"}>
              <span className="label step-idx">{String(i + 1).padStart(2, "0")} / 07</span>
              <h3 className="h3 step-title">{s.title}</h3>
              <p className="body-s step-body">{s.body}</p>
              <ul className="step-facts">
                {s.facts.map((f) => (
                  <li key={f} className="label">
                    {f}
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
