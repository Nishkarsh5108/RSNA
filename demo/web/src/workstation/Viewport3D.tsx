import { useEffect, useMemo, useRef, useState } from "react";
import type { PipelineEvent } from "../lib/api";
import { FINDINGS } from "../lib/findings";
import { Scene3D, type SceneBox, type SceneState, type SeriesView } from "./Scene3D";
import { boxSlices, getState, overlayMap, pipelineView, setState, useStore, type PipelineView, type State } from "./store";

/** Box z (detector convention) -> the unit cube's w over the slices the classifier actually includes. */
function boxToUnit(b: number[], native: number): number[] {
  const zr = boxSlices(b, native);
  const w0 = zr ? zr[0] / native : (b[2] * (native - 1)) / native;
  const w1 = zr ? (zr[1] + 1) / native : (b[5] * (native - 1) + 1) / native;
  return [b[0], b[1], w0, b[3], b[4], w1];
}

function buildScene(s: State, view: PipelineView, replaying: boolean, cut: boolean, lo: number): SceneState {
  const series: Record<string, SeriesView> = {};
  const sum = s.run.summary;
  const live = s.run.state === "running" || replaying;
  const f = s.finding;
  // global maximum of the shown quantity for the selected finding, so series compare honestly
  let gmax = s.overlay === "lesion" ? 1 : 1e-9; // lesion head: absolute probability
  if (f !== null && !live && s.overlay !== "lesion") {
    for (const ex of Object.values(s.explain)) {
      const src = overlayMap(ex, s.overlay) ?? ex.contrib;
      const cells = ex.h * ex.w;
      for (let k = 0; k < ex.d; k++) for (let i = 0; i < cells; i++) gmax = Math.max(gmax, src[(k * ex.k + f) * cells + i]);
    }
  }
  for (const m of s.manifest?.series ?? []) {
    const uid = m.uid;
    const g = s.geos[uid];
    if (!g || !m.include) {
      series[uid] = { visible: false, emphasis: 0, slice: 0, showSlice: false, boxes: [], crop: null, cropOpacity: 0, cells: null, lattice: null };
      continue;
    }
    const native = g.depth;
    const res = sum?.series[uid];
    const focus = live && view.focus === uid;
    const emphasis = live ? (focus ? 1 : 0.18) : uid === s.active ? 0.55 : 0.22;
    const boxes: SceneBox[] = [];
    const det = view.boxes[uid];
    if (live && det && view.stage === "detector") {
      if (focus || view.focus === null) det.mean.forEach((b) => boxes.push({ box: boxToUnit(b, native), color: focus ? "accent" : "white", opacity: focus ? 0.75 : 0.28 }));
      if (focus) {
        det.direct.forEach((b) => boxes.push({ box: boxToUnit(b, native), color: "white", opacity: 0.22, dashed: true }));
        det.mirrored.forEach((b) => boxes.push({ box: boxToUnit(b, native), color: "white", opacity: 0.22, dashed: true }));
      }
    } else if (live && det && (view.stage === "classifier" || view.stage === "ensemble") && focus) {
      det.mean.forEach((b) => boxes.push({ box: boxToUnit(b, native), color: "white", opacity: 0.12 }));
    } else if (!live && res && f !== null) {
      if (uid === s.active) boxes.push({ box: boxToUnit(res.boxes[f], native), color: "accent", opacity: 0.95 });
      else boxes.push({ box: boxToUnit(res.boxes[f], native), color: "white", opacity: 0.12 });
    } else if (!live && res && f === null && s.overlay === "box") {
      res.boxes.forEach((b) => boxes.push({ box: boxToUnit(b, native), color: "white", opacity: 0.3 }));
    }
    const fr = view.frames[uid] ?? (res?.crop ? { crop: res.crop, mirror: !!res.mirror, depth: res.depth!, native: res.native! } : null);
    const ws = res?.slicePos?.map((p) => (p + 0.5) / native) ?? (fr ? Array.from({ length: fr.depth }, (_, k) => ((fr.native > fr.depth ? (k * (fr.native - 1)) / (fr.depth - 1) : k) + 0.5) / native) : []);
    let cropOpacity = 0;
    let lattice: SeriesView["lattice"] = null;
    if (fr) {
      if (live && focus && view.stage === "classifier") {
        cropOpacity = 0.9;
        const enc = view.ev?.kind === "model" || view.ev?.kind === "series";
        lattice = { u0: fr.crop[0], v0: fr.crop[1], u1: fr.crop[2], v1: fr.crop[3], ws, h: 8, w: 8, opacity: enc ? 0.07 : 0.025 };
      } else if (!live && s.overlay === "box") cropOpacity = 0.45;
    }
    let cells: SeriesView["cells"] = null;
    const ex = s.explain[uid];
    // evidence is drawn for one series at a time (the active one, or the one being processed), never stacked
    const showCells = f !== null && ex && res?.crop && (live ? focus && view.ev?.kind === "series" : uid === s.active);
    if (showCells && s.overlay !== "box" && overlayMap(ex, s.overlay)) {
      const src = overlayMap(ex, s.overlay)!;
      const n = ex.h * ex.w;
      const alpha = new Float32Array(ex.d * n);
      let lmax = 1e-9;
      if (live && s.overlay !== "lesion") for (let k = 0; k < ex.d; k++) for (let i = 0; i < n; i++) lmax = Math.max(lmax, src[(k * ex.k + f!) * n + i]);
      if (s.overlay === "lesion") lmax = 1;
      const norm = live ? lmax : gmax;
      for (let k = 0; k < ex.d; k++)
        for (let i = 0; i < n; i++) {
          const v = src[(k * ex.k + f!) * n + i] / norm;
          alpha[k * n + i] = s.overlay === "lesion" ? (v > 0.06 ? 0.75 * Math.pow(v, 0.8) : 0) : v > 0.06 ? 0.62 * Math.pow(v, 1.35) : 0;
        }
      cells = { u0: res!.crop![0], v0: res!.crop![1], u1: res!.crop![2], v1: res!.crop![3], ws, alpha, h: ex.h, w: ex.w };
    }
    series[uid] = { visible: true, emphasis, slice: s.slice[uid] ?? 0, showSlice: true, boxes, crop: fr?.crop ?? null, cropOpacity, cells, lattice };
  }
  return { active: live && view.focus ? view.focus : s.active, mode: s.render, cut, window: [lo, 0.96], series, point: live ? null : s.point, cellColor: s.overlay === "attention" ? "white" : "accent" };
}

function caption(ev: PipelineEvent | null, s: State, view: PipelineView): { step: string; title: string; detail: string } | null {
  if (!ev) return null;
  const meta = typeof ev.series === "string" ? s.manifest?.series.find((x) => x.uid === ev.series) : undefined;
  const name = meta ? `${meta.plane.toUpperCase()} · ${(meta.description || meta.uid.slice(-6)).slice(0, 24)}` : "";
  const n = s.manifest?.series.filter((x) => x.include).length ?? 0;
  const models = s.status?.models?.classifiers ?? [];
  const mi = models.find((m) => m.id === ev.model);
  switch (`${ev.stage}:${ev.kind}`) {
    case "pipeline:start":
      return { step: "00", title: "Pipeline started", detail: `${n} series · ${models.length} classifier checkpoints · ${s.status?.gpu?.name ?? "GPU"}` };
    case "side:start":
      return { step: "01", title: "Knee side", detail: "ResNet-18d on every coronal and axial series" };
    case "side:series":
      return { step: "01", title: "Knee side", detail: `${name} · p(right) ${ev.pRight?.toFixed(3)} · ${ev.slices} slices at 224 px · ${ev.ms?.toFixed(0)} ms` };
    case "side:done":
      return { step: "01", title: `${(ev.knee ?? "").toUpperCase()} knee`, detail: ev.byDefault ? "no coronal/axial series: right by default" : `mean p(right) ${ev.pRight?.toFixed(3)}${ev.knee === "left" ? " · coronal and axial series will be mirrored" : ""}` };
    case "detector:start":
      return { step: "02", title: "ROI detector", detail: "ConvNeXt-T per slice, 3D top-down fusion, 12 boxes per series" };
    case "detector:series":
      return { step: "02", title: "ROI detector", detail: `${name} · input ${ev.input?.join("×")} · 2 passes (direct + mirrored) averaged · ${ev.ms?.toFixed(0)} ms` };
    case "detector:done":
      return { step: "02", title: "12 boxes per series", detail: ev.fallback?.length ? `${ev.fallback.length} series used the plane's median box` : "every series has its own boxes" };
    case "classifier:start":
      return { step: "03", title: "Box-pooling classifiers", detail: `${models.length} checkpoints, each on every series` };
    case "classifier:frame":
      return { step: "03", title: "Frame and crop", detail: `${name} · 140 mm field at 384 px → 112 mm crop at 256 px · ${ev.depth} slices${ev.mirror ? " · mirrored" : ""}` };
    case "classifier:model":
      return {
        step: "04",
        title: `Encode + pool · ${ev.model}`,
        detail: `${mi ? `${mi.labels}, fold ${mi.fold} · ` : ""}${ev.slices} slices × ${ev.cells?.join("×")} cells × ${ev.channels} ch → 12 box pools → 12 logits · ${ev.ms?.toFixed(0)} ms`,
      };
    case "classifier:series":
      return { step: "04", title: "Series result", detail: `${name} · mean of ${models.length} models` };
    case "ensemble:done":
      return { step: "05", title: "Ensemble", detail: `mean over series, then over models · ${ev.seconds?.toFixed(1)} s · peak VRAM ${ev.peakVramGB?.toFixed(2)} GB` };
    default:
      if (ev.kind === "error") return { step: "!!", title: "Series skipped", detail: `${name} · ${ev.error}` };
      return null;
  }
}

export default function Viewport3D() {
  const host = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<Scene3D | null>(null);
  const gizmo = useRef<SVGSVGElement>(null);
  const [cut, setCut] = useState(false);
  const [lo, setLo] = useState(0.22);
  const manifest = useStore((s) => s.manifest);
  const geos = useStore((s) => s.geos);
  const volumes = useStore((s) => s.volumes);
  const full = useStore((s) => s);
  const view = useMemo(() => pipelineView(full.run.events, full.cursor), [full.run.events, full.cursor]);
  const replaying = full.run.events.length > 0 && full.cursor < full.run.events.length - 1 && full.run.state !== "running";

  useEffect(() => {
    const sc = new Scene3D(host.current!);
    sceneRef.current = sc;
    if (import.meta.env.DEV) (window as any).__scene = sc;
    const axes: [string, [number, number, number]][] = [["L", [1, 0, 0]], ["P", [0, 1, 0]], ["S", [0, 0, 1]], ["R", [-1, 0, 0]], ["A", [0, -1, 0]], ["I", [0, 0, -1]]];
    sc.onFrame = () => {
      const svg = gizmo.current;
      if (!svg) return;
      axes.forEach(([, v], i) => {
        const [x, y] = sc.projectAxis(v);
        const line = svg.children[i] as SVGLineElement;
        const text = svg.children[i + 6] as SVGTextElement;
        line.setAttribute("x2", String(30 + x * 20));
        line.setAttribute("y2", String(30 + y * 20));
        text.setAttribute("x", String(30 + x * 26));
        text.setAttribute("y", String(30 + y * 26 + 3));
      });
    };
    return () => sc.dispose();
  }, []);

  useEffect(() => {
    if (manifest) sceneRef.current?.setStudy(manifest.series, geos);
  }, [manifest?.sid, geos]);

  useEffect(() => {
    const sc = sceneRef.current;
    if (!sc) return;
    for (const [uid, v] of Object.entries(volumes)) sc.setVolume(uid, v);
  }, [volumes]);

  useEffect(() => {
    const sc = sceneRef.current;
    if (sc && manifest) sc.apply(buildScene(getState(), view, replaying, cut, lo));
  }, [full, view, replaying, cut, lo, manifest]);

  // replay: step back to the whole study
  useEffect(() => {
    if (replaying) sceneRef.current?.home();
  }, [replaying]);

  // fly to the selected finding's box in the active series
  const sum = full.run.summary;
  useEffect(() => {
    const s = getState();
    if (full.finding === null || !sum || !s.active || replaying) return;
    const r = sum.series[s.active];
    const g = s.geos[s.active];
    if (r && g) sceneRef.current?.focusBox(s.active, boxToUnit(r.boxes[full.finding], g.depth));
  }, [full.finding, sum, replaying]);

  const cap = caption(view.ev, full, view);
  const showCap = full.run.state === "running" || replaying;
  const shownUid = (full.run.state === "running" || replaying) && view.focus ? view.focus : full.active;
  const active = full.manifest?.series.find((x) => x.uid === shownUid);
  return (
    <div className="vp3">
      <div className="vp3-canvas" ref={host} />
      <div className="vp-head vp3-head">
        <span className="label label--fg">3D · PATIENT SPACE</span>
        <span className="label">{active ? `volume: ${active.plane.toLowerCase()} · ${(active.description || "").slice(0, 26)}` : ""}</span>
        <div className="vp-tools">
          {(["dvr", "mip", "slices"] as const).map((m) => (
            <button key={m} className={`tool ${full.render === m ? "on" : ""}`} onClick={() => setState({ render: m })}>
              {m === "dvr" ? "Volume" : m === "mip" ? "MIP" : "Slices"}
            </button>
          ))}
          <span className="tool-sep" />
          <label className="thr label" title="Lower threshold of the volume rendering (display values)">
            thr
            <input type="range" min={0} max={0.8} step={0.01} value={lo} onChange={(e) => setLo(+e.target.value)} />
          </label>
          <button className={`tool ${cut ? "on" : ""}`} onClick={() => setCut((c) => !c)} title="Cut the volume at the current slice">
            Cut
          </button>
          <button className="tool" onClick={() => sceneRef.current?.home()}>
            Reset
          </button>
        </div>
      </div>
      <svg className="gizmo" ref={gizmo} width="60" height="60" viewBox="0 0 60 60">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <line key={i} x1="30" y1="30" x2="30" y2="30" stroke={i < 3 ? "rgba(242,241,236,.6)" : "rgba(242,241,236,.18)"} strokeWidth="1" />
        ))}
        {["L", "P", "S", "R", "A", "I"].map((t, i) => (
          <text key={t} x="30" y="30" textAnchor="middle" fontSize="8.5" fontFamily="IBM Plex Mono" fill={i < 3 ? "rgba(242,241,236,.8)" : "rgba(242,241,236,.3)"}>
            {t}
          </text>
        ))}
      </svg>
      {showCap && cap && (
        <div className="stagecap" key={`${cap.step}${cap.title}`}>
          <span className="stagecap-step num">{cap.step}</span>
          <div>
            <div className="stagecap-title">{cap.title}</div>
            <div className="stagecap-detail mono">{cap.detail}</div>
          </div>
          <span className="stagecap-t num">{(view.t / 1000).toFixed(2)} s</span>
        </div>
      )}
      {!showCap && full.finding !== null && sum && (
        <div className="legend3d mono">
          <span className="sw sw-box" /> {FINDINGS[full.finding].short} ROI box
          {full.overlay !== "box" && (
            <>
              <span className={`sw sw-cell ${full.overlay === "attention" ? "white" : ""}`} />{" "}
              {full.overlay === "evidence"
                ? "positive evidence per feature cell (14 mm)"
                : full.overlay === "attention"
                  ? "pooling weight per feature cell (14 mm)"
                  : "lesion head: predicted lesion coverage per cell (14 mm)"}
            </>
          )}
        </div>
      )}
      {!showCap && <div className="hint3d label">drag rotate · right-drag pan · wheel zoom</div>}
    </div>
  );
}
