import { useEffect } from "react";
import { PROJECT } from "../lib/findings";
import { go } from "../lib/route";
import "../styles/workstation.css";
import { DropZone, FindingsPanel, PipelinePanel, SeriesPanel } from "./Panels";
import { clearStudy, init, setState, useStore } from "./store";
import Viewport2D from "./Viewport2D";
import Viewport3D from "./Viewport3D";

function TopBar() {
  const status = useStore((s) => s.status);
  const m = useStore((s) => s.manifest);
  const layout = useStore((s) => s.layout);
  const n = status?.models?.classifiers.length ?? 0;
  return (
    <header className="ws-top">
      <button className="ws-mark" onClick={() => go("/")} title="Back to the site">
        <span className="mark-sq">
          <span />
        </span>
        <span className="ws-name">{PROJECT.name}</span>
        <span className="ws-slash">/</span>
        <span className="label label--fg">Workstation</span>
      </button>
      <div className="ws-study mono">
        {m ? (
          <>
            <span className="label">study</span> {m.shortId} <span className="sep">·</span> {m.source} <span className="sep">·</span> {m.series.filter((s) => s.include).length} series
          </>
        ) : (
          <span className="label">no study loaded</span>
        )}
      </div>
      <div className="ws-right">
        {m && (
          <div className="seg seg-top">
            <button className={`tool ${layout === "quad" ? "on" : ""}`} onClick={() => setState({ layout: "quad" })}>
              3D + planes
            </button>
            <button className={`tool ${layout === "focus3d" ? "on" : ""}`} onClick={() => setState({ layout: "focus3d" })}>
              3D
            </button>
          </div>
        )}
        <div className="ws-dev mono" title={status?.error ?? ""}>
          <span className={`dot ${status?.ready ? "dot--accent" : status?.error ? "" : "dot--live"}`} style={!status?.ready && status?.error ? { background: "var(--danger)" } : undefined} />
          {status?.ready
            ? `${status.gpu?.name.replace("NVIDIA GeForce ", "") ?? status.device} · ${status.gpu?.totalGB.toFixed(1)} GB · ${n} models`
            : status?.error
              ? "server offline"
              : "loading models"}
        </div>
        {m && (
          <button className="btn btn--sm" onClick={() => clearStudy()} title="Delete this study from the local cache">
            Purge study
          </button>
        )}
      </div>
    </header>
  );
}

export default function Workstation() {
  const m = useStore((s) => s.manifest);
  const layout = useStore((s) => s.layout);
  const loading = useStore((s) => s.loading);
  useEffect(() => {
    init();
    document.title = `Workstation — ${PROJECT.name}`;
    return () => {
      document.title = PROJECT.name;
    };
  }, []);
  return (
    <div className="ws">
      <TopBar />
      <aside className="ws-left scroll-y">
        {m ? (
          <>
            <SeriesPanel />
            <PipelinePanel />
          </>
        ) : (
          <section className="panel">
            <header className="panel-h">
              <span className="label label--fg">Pipeline</span>
            </header>
            <ol className="intro-steps body-s">
              <li><span className="num">01</span> Knee side from the coronal and axial series</li>
              <li><span className="num">02</span> One 3D box per finding per series</li>
              <li><span className="num">03</span> 112 mm crop, 8 × 8 feature cells per slice</li>
              <li><span className="num">04</span> Each finding pools only its own box</li>
              <li><span className="num">05</span> Mean over series, then over models</li>
            </ol>
          </section>
        )}
      </aside>
      <main className={`ws-main ${layout}`}>
        {m ? (
          <>
            <div className="vp-3d-cell">
              <Viewport3D />
            </div>
            {layout === "quad" && (
              <div className="vp-2d-row">
                <Viewport2D plane="Sagittal" />
                <Viewport2D plane="Coronal" />
                <Viewport2D plane="Axial" />
              </div>
            )}
            {loading.phase === "volumes" && (
              <div className="vol-progress">
                <span style={{ width: `${loading.progress * 100}%` }} />
              </div>
            )}
          </>
        ) : (
          <DropZone />
        )}
      </main>
      <aside className="ws-right-col scroll-y">
        <FindingsPanel />
        <p className="disclaimer label">Research prototype. Not a medical device; not for diagnosis.</p>
      </aside>
    </div>
  );
}
