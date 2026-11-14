// A native-plane slice viewer. Overlays are drawn in the series' normalized coordinates, exactly where the detector put
// its boxes and where the classifier's feature cells lie. "Model input" shows the 256 x 256 crop the classifier read.
import { useEffect, useMemo, useRef, useState } from "react";
import { FINDINGS } from "../lib/findings";
import { edgeLetters, sliceLocation, toPatient, toUnit, type Geo } from "./geometry";
import { boxSlices, pipelineView, setPoint, setSlice, setState, sliceProfile, useStore, type Vol } from "./store";

type Plane = "Sagittal" | "Coronal" | "Axial";

interface ViewXf {
  zoom: number;
  panX: number;
  panY: number;
  wc: number; // window centre on the display values (0..255)
  ww: number;
}
const DEFAULT_XF: ViewXf = { zoom: 1, panX: 0, panY: 0, wc: 128, ww: 256 };

function sliceImage(vol: Vol, k: number, wc: number, ww: number, mirror = false): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = vol.w;
  c.height = vol.h;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(vol.w, vol.h);
  const off = k * vol.w * vol.h;
  const lo = wc - ww / 2;
  const sc = 255 / Math.max(1, ww);
  for (let y = 0; y < vol.h; y++) {
    for (let x = 0; x < vol.w; x++) {
      const v = vol.data[off + y * vol.w + (mirror ? vol.w - 1 - x : x)];
      const g = Math.max(0, Math.min(255, (v - lo) * sc));
      const i = (y * vol.w + x) * 4;
      img.data[i] = g;
      img.data[i + 1] = g;
      img.data[i + 2] = g * 0.985;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export default function Viewport2D({ plane }: { plane: Plane }) {
  const uid = useStore((s) => s.planeSeries[plane]);
  const manifest = useStore((s) => s.manifest);
  const meta = manifest?.series.find((x) => x.uid === uid);
  const geo = useStore((s) => (uid ? s.geos[uid] : undefined));
  const vol = useStore((s) => (uid ? s.volumes[uid] : undefined));
  const input = useStore((s) => (uid ? s.inputs[uid] : undefined));
  const ex = useStore((s) => (uid ? s.explain[uid] : undefined));
  const k = useStore((s) => (uid ? s.slice[uid] ?? 0 : 0));
  const point = useStore((s) => s.point);
  const finding = useStore((s) => s.finding);
  const overlay = useStore((s) => s.overlay);
  const modelView = useStore((s) => s.modelView);
  const run = useStore((s) => s.run);
  const cursor = useStore((s) => s.cursor);
  const active = useStore((s) => s.active);
  const view = useMemo(() => pipelineView(run.events, cursor), [run.events, cursor]);
  const res = uid ? run.summary?.series[uid] : undefined;
  const frame = uid ? view.frames[uid] : undefined;
  const detBoxes = uid ? view.boxes[uid] : undefined;

  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [xf, setXf] = useState<ViewXf>(DEFAULT_XF);
  const [size, setSize] = useState({ w: 300, h: 300 });
  const [hover, setHover] = useState<{ x: number; y: number; mm: [number, number] } | null>(null);
  useEffect(() => setXf(DEFAULT_XF), [uid, modelView]);

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // classifier slice nearest to the shown native slice
  const ck = useMemo(() => {
    const sp = res?.slicePos;
    if (!sp) return -1;
    let best = 0;
    for (let i = 1; i < sp.length; i++) if (Math.abs(sp[i] - k) < Math.abs(sp[best] - k)) best = i;
    return best;
  }, [res, k]);

  const image = useMemo(() => {
    if (modelView && input && ck >= 0) return sliceImage(input, ck, xf.wc, xf.ww);
    if (!vol) return null;
    return sliceImage(vol, Math.min(k, vol.d - 1), xf.wc, xf.ww);
  }, [vol, input, modelView, ck, k, xf.wc, xf.ww]);

  const profile = useMemo(() => (ex && finding !== null ? sliceProfile(ex, finding) : null), [ex, finding]);

  // layout: the image rectangle in canvas pixels (CSS px)
  const layout = useMemo(() => {
    if (!geo) return null;
    const inModel = modelView && !!input && !!res?.crop;
    // model view: the crop is square in mm (112 mm); native: the image's physical extent
    const wmm = inModel ? 1 : geo.cols * geo.dc;
    const hmm = inModel ? 1 : geo.rows * geo.dr;
    const pad = 28;
    const s = Math.min((size.w - pad * 2) / wmm, (size.h - pad * 2 - 6) / hmm) * xf.zoom;
    const W = wmm * s,
      H = hmm * s;
    return { x: (size.w - W) / 2 + xf.panX, y: (size.h - H) / 2 + xf.panY, W, H, s, inModel };
  }, [geo, size, xf, modelView, input, res]);

  // ---------------------------------------------------------------- draw
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = Math.min(devicePixelRatio, 2);
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    if (!geo || !layout || !image) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, layout.x, layout.y, layout.W, layout.H);

    // native normalized (u, v) -> canvas; in model view, (u, v) are crop coordinates already
    const crop = res?.crop ?? frame?.crop;
    const mirror = res?.mirror ?? frame?.mirror ?? false;
    const P = (u: number, v: number): [number, number] => {
      if (layout.inModel && crop) {
        let cu = (u - crop[0]) / (crop[2] - crop[0]);
        const cv = (v - crop[1]) / (crop[3] - crop[1]);
        if (mirror) cu = 1 - cu;
        return [layout.x + cu * layout.W, layout.y + cv * layout.H];
      }
      return [layout.x + u * layout.W, layout.y + v * layout.H];
    };
    const rect = (u0: number, v0: number, u1: number, v1: number) => {
      const [a, b] = P(u0, v0);
      const [c2, d] = P(u1, v1);
      return [Math.min(a, c2), Math.min(b, d), Math.abs(c2 - a), Math.abs(d - b)] as const;
    };
    const native = geo.depth;
    const showStage = run.state === "running" || (run.events.length > 0 && cursor < run.events.length - 1);
    const final = !showStage; // results (evidence, finding box, crosshair) only once the trace is at its end

    // detector output while the trace is at the detector stage for this series (or before results exist)
    if (detBoxes && ((showStage && view.stage === "detector") || (!res && run.state !== "idle"))) {
      ctx.font = "500 9.5px 'IBM Plex Mono', monospace";
      const placed: [number, number, number][] = []; // label boxes already drawn (x, y, width)
      detBoxes.mean.forEach((b, i) => {
        const zr = boxSlices(b, native);
        if (!zr || k < zr[0] - 0.5 || k > zr[1] + 0.5) return;
        const [x, y, w, h] = rect(b[0], b[1], b[3], b[4]);
        ctx.strokeStyle = view.focus === uid ? "rgba(245,183,0,0.85)" : "rgba(242,241,236,0.55)";
        ctx.lineWidth = 1;
        ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
        // label at the box's top-left corner, pushed down until it clears the labels already drawn
        const tw = ctx.measureText(FINDINGS[i].short).width + 4;
        let ly = y + 10;
        while (placed.some(([px, py, pw]) => Math.abs(py - ly) < 10 && x + 3 < px + pw && px < x + 3 + tw) && ly < y + h - 2) ly += 10;
        placed.push([x + 3, ly, tw]);
        ctx.fillStyle = "rgba(242,241,236,0.85)";
        ctx.fillText(FINDINGS[i].short, x + 3, ly);
      });
    }

    // crop window: the 112 mm the classifier reads
    if (crop && !layout.inModel && (overlay === "box" || (showStage && view.stage === "classifier" && view.focus === uid))) {
      const [x, y, w, h] = rect(crop[0], crop[1], crop[2], crop[3]);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(242,241,236,0.5)";
      ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
      ctx.setLineDash([]);
    }

    // evidence / attention cells of the selected finding on the nearest classifier slice
    if (final && ex && res?.crop && finding !== null && ck >= 0 && overlay !== "box") {
      const cells = ex.h * ex.w;
      const base = (ck * ex.k + finding) * cells;
      const src = overlay === "evidence" ? ex.contrib : ex.share;
      let max = 1e-9;
      for (let s2 = 0; s2 < ex.d; s2++) for (let i = 0; i < cells; i++) max = Math.max(max, src[(s2 * ex.k + finding) * cells + i]);
      const [cx0, cy0, cx1, cy1] = res.crop;
      for (let r = 0; r < ex.h; r++) {
        for (let q = 0; q < ex.w; q++) {
          const val = src[base + r * ex.w + q];
          const a = Math.max(0, val / max);
          if (a < 0.02) continue;
          const u0 = cx0 + (q / ex.w) * (cx1 - cx0),
            u1 = cx0 + ((q + 1) / ex.w) * (cx1 - cx0);
          const v0 = cy0 + (r / ex.h) * (cy1 - cy0),
            v1 = cy0 + ((r + 1) / ex.h) * (cy1 - cy0);
          const [x, y, w, h] = rect(u0, v0, u1, v1);
          ctx.fillStyle = overlay === "evidence" ? `rgba(245,183,0,${(0.62 * Math.pow(a, 0.8)).toFixed(3)})` : `rgba(242,241,236,${(0.5 * a).toFixed(3)})`;
          ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
        }
      }
      // the 8 x 8 lattice, hairline
      const [gx, gy, gw, gh] = rect(cx0, cy0, cx1, cy1);
      ctx.strokeStyle = "rgba(242,241,236,0.10)";
      ctx.beginPath();
      for (let i = 0; i <= ex.w; i++) {
        const xx = Math.round(gx + (i / ex.w) * gw) + 0.5;
        ctx.moveTo(xx, gy);
        ctx.lineTo(xx, gy + gh);
      }
      for (let i = 0; i <= ex.h; i++) {
        const yy = Math.round(gy + (i / ex.h) * gh) + 0.5;
        ctx.moveTo(gx, yy);
        ctx.lineTo(gx + gw, yy);
      }
      ctx.stroke();
    }

    // selected finding's ROI box
    if (final && res && finding !== null) {
      const b = res.boxes[finding];
      const zr = boxSlices(b, native);
      const inside = !!zr && k >= zr[0] - 0.5 && k <= zr[1] + 0.5;
      const [x, y, w, h] = rect(b[0], b[1], b[3], b[4]);
      ctx.strokeStyle = inside ? "rgba(245,183,0,0.95)" : "rgba(245,183,0,0.25)";
      ctx.setLineDash(inside ? [] : [3, 4]);
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, Math.round(w), Math.round(h));
      ctx.setLineDash([]);
      // corner ticks
      ctx.fillStyle = inside ? "#f5b700" : "rgba(245,183,0,0.4)";
      ctx.font = "500 9.5px 'IBM Plex Mono', monospace";
      ctx.fillText(`${FINDINGS[finding].short} BOX${inside ? "" : " · OUTSIDE Z"}`, x + 1, y - 5);
    }

    // crosshair
    if (final && point && !layout.inModel) {
      const [u, v, w] = toUnit(geo, point);
      const onSlice = Math.abs(w * geo.depth - 0.5 - k) < 1.0;
      const [px, py] = P(u, v);
      ctx.strokeStyle = onSlice ? "rgba(245,183,0,0.75)" : "rgba(245,183,0,0.3)";
      ctx.lineWidth = 1;
      const gap = 7;
      ctx.beginPath();
      ctx.moveTo(layout.x, Math.round(py) + 0.5);
      ctx.lineTo(px - gap, Math.round(py) + 0.5);
      ctx.moveTo(px + gap, Math.round(py) + 0.5);
      ctx.lineTo(layout.x + layout.W, Math.round(py) + 0.5);
      ctx.moveTo(Math.round(px) + 0.5, layout.y);
      ctx.lineTo(Math.round(px) + 0.5, py - gap);
      ctx.moveTo(Math.round(px) + 0.5, py + gap);
      ctx.lineTo(Math.round(px) + 0.5, layout.y + layout.H);
      ctx.stroke();
    }
  }, [geo, layout, image, size, res, frame, ex, finding, overlay, ck, k, point, detBoxes, view, run.state, run.events.length, cursor, uid]);

  // ---------------------------------------------------------------- interaction
  const drag = useRef<{ x: number; y: number; b: number; xf: ViewXf; moved: boolean } | null>(null);
  const toUV = (cx: number, cy: number): [number, number] | null => {
    if (!layout) return null;
    return [(cx - layout.x) / layout.W, (cy - layout.y) / layout.H];
  };
  const onWheel = (e: React.WheelEvent) => {
    if (!uid || !geo) return;
    if (e.ctrlKey || e.shiftKey) {
      const f = Math.exp(-e.deltaY * 0.0015);
      setXf((x) => ({ ...x, zoom: Math.max(0.5, Math.min(8, x.zoom * f)) }));
    } else {
      setSlice(uid, k + (e.deltaY > 0 ? 1 : -1));
    }
  };
  const onDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, b: e.button, xf, moved: false };
  };
  const onMove = (e: React.PointerEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    const uv = toUV(e.clientX - r.left, e.clientY - r.top);
    if (uv && geo && !layout?.inModel) setHover({ x: e.clientX - r.left, y: e.clientY - r.top, mm: [uv[0] * geo.cols * geo.dc, uv[1] * geo.rows * geo.dr] });
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x,
      dy = e.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (!d.moved) return;
    if (d.b === 2) setXf({ ...d.xf, wc: d.xf.wc - dy * 0.6, ww: Math.max(8, d.xf.ww + dx * 0.9) });
    else setXf({ ...d.xf, panX: d.xf.panX + dx, panY: d.xf.panY + dy });
  };
  const onUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (!d || d.moved || d.b !== 0 || !geo || !uid || layout?.inModel) return;
    const r = wrap.current!.getBoundingClientRect();
    const uv = toUV(e.clientX - r.left, e.clientY - r.top);
    if (!uv || uv[0] < 0 || uv[0] > 1 || uv[1] < 0 || uv[1] > 1) return;
    setPoint(toPatient(geo, uv[0], uv[1], (k + 0.5) / geo.depth), uid);
    if (active !== uid) setState({ active: uid });
  };

  // ---------------------------------------------------------------- chrome
  const letters = geo ? edgeLetters(geo) : null;
  const planeSeries = manifest?.series.filter((s) => s.plane === plane && s.include) ?? [];
  const zr = res && finding !== null ? boxSlices(res.boxes[finding], geo?.depth ?? 1) : null;
  const isFocus = view.focus === uid && (run.state === "running" || cursor < run.events.length - 1);
  return (
    <div className={`vp2 ${isFocus ? "is-focus" : ""} ${active === uid ? "is-active" : ""}`}>
      <div className="vp-head">
        <span className="label label--fg">{plane.slice(0, 3).toUpperCase()}</span>
        {planeSeries.length > 1 ? (
          <select className="vp-select mono" value={uid} onChange={(e) => setState((s) => ({ planeSeries: { ...s.planeSeries, [plane]: e.target.value } }))}>
            {planeSeries.map((s) => (
              <option key={s.uid} value={s.uid}>
                {(s.description || s.uid.slice(-6)).slice(0, 26)}
              </option>
            ))}
          </select>
        ) : (
          <span className="label">{meta ? (meta.description || meta.uid.slice(-6)).slice(0, 28) : "no series"}</span>
        )}
        <span className="vp-head-r label">{meta ? (meta.fatSuppression ? "FS" : "NON-FS") : ""}</span>
      </div>
      <div
        className="vp-canvas"
        ref={wrap}
        onWheel={onWheel}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => setXf(DEFAULT_XF)}
        onContextMenu={(e) => e.preventDefault()}
      >
        <canvas ref={canvas} />
        {!vol && uid && <div className="vp-empty label">Loading volume…</div>}
        {!uid && <div className="vp-empty label">No {plane.toLowerCase()} series</div>}
        {letters && !layout?.inModel && (
          <>
            <span className="ol ol-t">{letters[0]}</span>
            <span className="ol ol-r">{letters[1]}</span>
            <span className="ol ol-b">{letters[2]}</span>
            <span className="ol ol-l">{letters[3]}</span>
          </>
        )}
        {geo && meta && (
          <>
            <div className="corner c-tl mono">
              {layout?.inModel ? (
                <>
                  <span className="accent">MODEL INPUT</span> 256 × 256 · 112 mm
                  <br />
                  slice {ck + 1}/{res?.slicePos?.length} {res?.mirror ? "· mirrored (left knee)" : ""}
                </>
              ) : (
                <>
                  {meta.rows}×{meta.cols} · {meta.pixelSpacing[1].toFixed(2)} mm
                  <br />
                  TR {meta.tr?.toFixed(0) ?? "–"} · TE {meta.te?.toFixed(0) ?? "–"}
                </>
              )}
            </div>
            <div className="corner c-br mono">
              Im {k + 1}/{geo.depth}
              <br />
              {sliceLocation(geo, k).toFixed(1)} mm
            </div>
            <div className="corner c-bl mono">
              W {Math.round(xf.ww)} L {Math.round(xf.wc)} · ×{xf.zoom.toFixed(2)}
              {hover && <> · {hover.mm[0].toFixed(0)},{hover.mm[1].toFixed(0)} mm</>}
            </div>
          </>
        )}
        {geo && (
          <Scrubber
            depth={geo.depth}
            k={k}
            zr={zr}
            profile={profile?.pos ?? null}
            slicePos={res?.slicePos ?? null}
            onSet={(i) => uid && setSlice(uid, i)}
          />
        )}
      </div>
    </div>
  );
}

function Scrubber({ depth, k, zr, profile, slicePos, onSet }: { depth: number; k: number; zr: [number, number] | null; profile: Float32Array | null; slicePos: number[] | null; onSet: (k: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const set = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    onSet(Math.round(((e.clientY - r.top) / r.height) * depth - 0.5));
  };
  const max = profile ? Math.max(1e-9, ...profile) : 1;
  return (
    <div
      className="scrub"
      ref={ref}
      onPointerDown={(e) => {
        e.stopPropagation();
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        set(e);
      }}
      onPointerMove={(e) => e.buttons && (e.stopPropagation(), set(e))}
      onPointerUp={(e) => e.stopPropagation()}
      onWheel={(e) => e.stopPropagation()}
    >
      {zr && <div className="scrub-z" style={{ top: `${(zr[0] / depth) * 100}%`, height: `${((zr[1] - zr[0] + 1) / depth) * 100}%` }} />}
      {profile &&
        slicePos &&
        Array.from(profile).map((v, i) =>
          v > 0 ? <div key={i} className="scrub-bar" style={{ top: `${((slicePos[i] + 0.5) / depth) * 100}%`, width: `${4 + 14 * (v / max)}px` }} /> : null,
        )}
      <div className="scrub-k" style={{ top: `${((k + 0.5) / depth) * 100}%` }} />
    </div>
  );
}
