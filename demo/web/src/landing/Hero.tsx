import { useEffect, useRef } from "react";
import { Ferrofluid } from "./ferrofluid";
import { modelCount, type ServerState } from "./useStatus";

export default function Hero({ state }: { state: ServerState | null }) {
  const host = useRef<HTMLDivElement>(null);
  const readout = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!host.current) return;
    let field: Ferrofluid | null = null;
    try {
      field = new Ferrofluid(host.current);
    } catch {
      host.current.classList.add("hero-field--static"); // no WebGL: the type carries the hero alone
      return;
    }
    let lastWrite = 0;
    field.onFrame = (r) => {
      const now = performance.now();
      if (now - lastWrite < 90 || !readout.current) return;
      lastWrite = now;
      const f = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
      readout.current.textContent = `X ${f(r.x)}  Z ${f(r.z)}  H ${r.strength.toFixed(2)}  ${Math.round(r.fps)} FPS`;
    };
    return () => field?.dispose();
  }, []);

  const n = modelCount(state);
  return (
    <section className="hero" id="top">
      <div className="hero-field" ref={host} aria-hidden="true" />
      <div className="hero-top">
        <span className="label">RSNA 2026 · knee MRI · post-competition build</span>
        <span className="label hero-fig">
          FIG. 00 — Ferrofluid under a magnet: peaks form a hexagonal lattice. The cursor is the magnet.
          <br />
          <span ref={readout} className="hero-readout">X +0.00 Z +0.00 H 0.00</span>
        </span>
      </div>
      <div className="hero-body">
        <h1 className="display hero-title" aria-label="Knee Abnormality Detection">
          <span className="line"><span>Knee</span></span>
          <span className="line"><span>Abnormality</span></span>
          <span className="line"><span>Detection</span></span>
        </h1>
        <div className="hero-side">
          <p className="hero-desc">
            Reads a knee MRI study and estimates the probability of twelve findings: ligament and meniscal tears,
            osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture. Every step runs on this
            machine's GPU.
          </p>
          <div className="hero-actions">
            <a className="btn btn--accent hero-cta" href="#/workstation">
              Open the workstation <span className="arrow">→</span>
            </a>
            <span className="label">Drop a DICOM study, or load a local sample</span>
          </div>
        </div>
      </div>
      <dl className="hero-strip">
        <div>
          <dt className="label">Findings</dt>
          <dd className="num">12</dd>
        </div>
        <div>
          <dt className="label">Networks</dt>
          <dd className="num">3</dd>
        </div>
        <div>
          <dt className="label">Ensemble</dt>
          <dd className="num">{n ?? "–"} models</dd>
        </div>
        <div>
          <dt className="label">Macro ROC-AUC</dt>
          <dd className="num">
            0.958 <span className="label">private LB, late submission</span>
          </dd>
        </div>
        <div>
          <dt className="label">Inference</dt>
          <dd className="num">127.0.0.1</dd>
        </div>
      </dl>
    </section>
  );
}
