// Workstation state: one study, its volumes, the recorded pipeline trace and the user's view.
import { useSyncExternalStore } from "react";
import { api, type Manifest, type PipelineEvent, type Sample, type SeriesMeta, type Status, type Summary } from "../lib/api";
import { geometry, sliceOfW, toUnit, type Geo, type V3 } from "./geometry";

export interface Vol {
  data: Uint8Array;
  d: number;
  h: number;
  w: number;
}
export interface Explain {
  d: number;
  k: number;
  h: number;
  w: number;
  share: Float32Array;
  contrib: Float32Array;
  boxWeight: Float32Array;
}
export type Overlay = "evidence" | "attention" | "box";
export type Layout = "quad" | "focus3d";
export type RenderMode = "dvr" | "mip" | "slices";

export interface State {
  status: Status | null;
  samples: Sample[];
  manifest: Manifest | null;
  geos: Record<string, Geo>;
  volumes: Record<string, Vol>;
  inputs: Record<string, Vol>;
  explain: Record<string, Explain>;
  loading: { phase: "idle" | "reading" | "uploading" | "parsing" | "volumes"; progress: number; label: string };
  error: string | null;
  run: { state: "idle" | "running" | "done" | "error"; events: PipelineEvent[]; summary: Summary | null; error: string | null; startedAt: number };
  cursor: number; // index into run.events shown by the views (live: the last one)
  follow: boolean; // cursor follows new events
  playing: boolean;
  active: string | null; // series shown as a volume in 3D
  planeSeries: Record<string, string>; // plane -> series in that 2D viewport
  slice: Record<string, number>; // series -> slice index
  point: V3 | null; // crosshair (patient mm)
  finding: number | null;
  overlay: Overlay;
  modelView: boolean; // 2D viewports show the classifier's own input crop
  layout: Layout;
  render: RenderMode;
}

const initial: State = {
  status: null,
  samples: [],
  manifest: null,
  geos: {},
  volumes: {},
  inputs: {},
  explain: {},
  loading: { phase: "idle", progress: 0, label: "" },
  error: null,
  run: { state: "idle", events: [], summary: null, error: null, startedAt: 0 },
  cursor: -1,
  follow: true,
  playing: false,
  active: null,
  planeSeries: {},
  slice: {},
  point: null,
  finding: null,
  overlay: "evidence",
  modelView: false,
  layout: "quad",
  render: "dvr",
};

let state: State = initial;
const listeners = new Set<() => void>();
export const getState = () => state;
export function setState(patch: Partial<State> | ((s: State) => Partial<State>)) {
  state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
  listeners.forEach((l) => l());
}
const subscribe = (l: () => void) => (listeners.add(l), () => listeners.delete(l));
export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(subscribe, () => sel(state));
}

// ------------------------------------------------------------------------------------------- derived pipeline view
export interface PipelineView {
  ev: PipelineEvent | null;
  stage: PipelineEvent["stage"] | "ingest";
  focus: string | null; // series being processed at the cursor
  model: string | null;
  side: Record<string, number>;
  knee: "left" | "right" | null;
  boxes: Record<string, { mean: number[][]; direct: number[][]; mirrored: number[][]; depth: number; ms: number }>;
  frames: Record<string, { crop: number[]; mirror: boolean; depth: number; native: number }>;
  modelProbs: Record<string, Record<string, number[]>>; // series -> model -> probs
  seriesProbs: Record<string, number[]>;
  done: boolean;
  t: number;
}

export function pipelineView(events: PipelineEvent[], cursor: number): PipelineView {
  const v: PipelineView = { ev: null, stage: "ingest", focus: null, model: null, side: {}, knee: null, boxes: {}, frames: {}, modelProbs: {}, seriesProbs: {}, done: false, t: 0 };
  for (let i = 0; i <= cursor && i < events.length; i++) {
    const e = events[i];
    v.ev = e;
    v.stage = e.stage;
    v.t = e.t;
    const uid = typeof e.series === "string" ? e.series : null;
    v.focus = uid;
    v.model = e.model ?? null;
    if (e.stage === "side" && e.kind === "series" && uid) v.side[uid] = e.pRight ?? 0;
    if (e.stage === "side" && e.kind === "done") v.knee = e.knee ?? null;
    if (e.stage === "detector" && e.kind === "series" && uid)
      v.boxes[uid] = { mean: e.boxes!, direct: e.direct!, mirrored: e.mirrored!, depth: e.depth!, ms: e.ms ?? 0 };
    if (e.stage === "classifier" && e.kind === "frame" && uid) v.frames[uid] = { crop: e.crop!, mirror: !!e.mirror, depth: e.depth!, native: e.native! };
    if (e.stage === "classifier" && e.kind === "model" && uid) (v.modelProbs[uid] ??= {})[e.model!] = e.probs!;
    if (e.stage === "classifier" && e.kind === "series" && uid) v.seriesProbs[uid] = e.probs!;
    if (e.stage === "ensemble") v.done = true;
  }
  return v;
}

/** Native slice range of a finding's box in a series, as the classifier reads it (slice centres inside z0..z1). */
export function boxSlices(box: number[], native: number): [number, number] | null {
  const d = Math.min(native, 64);
  const pos = (k: number) => (native > d ? (k * (native - 1)) / (d - 1) : k);
  let lo = -1,
    hi = -1;
  for (let k = 0; k < d; k++) {
    const z = (k + 0.5) / d;
    if (z >= box[2] && z <= box[5]) {
      if (lo < 0) lo = k;
      hi = k;
    }
  }
  return lo < 0 ? null : [pos(lo), pos(hi)];
}

/** Per classifier slice: summed positive contribution and share for one finding. */
export function sliceProfile(ex: Explain, f: number) {
  const cells = ex.h * ex.w;
  const pos = new Float32Array(ex.d);
  const share = new Float32Array(ex.d);
  for (let s = 0; s < ex.d; s++) {
    const base = (s * ex.k + f) * cells;
    for (let c = 0; c < cells; c++) {
      const v = ex.contrib[base + c];
      if (v > 0) pos[s] += v;
      share[s] += ex.share[base + c];
    }
  }
  return { pos, share };
}

// ------------------------------------------------------------------------------------------- actions
let statusTimer: number | undefined;
export async function init() {
  const poll = async () => {
    try {
      setState({ status: await api.status() });
    } catch {
      setState({ status: { ready: false, error: "local server not reachable on 127.0.0.1:8000", host: "127.0.0.1:8000" } });
    }
  };
  await poll();
  clearInterval(statusTimer);
  statusTimer = window.setInterval(poll, 4000);
  try {
    setState({ samples: await api.samples() });
  } catch {
    /* offline: the drop zone says so */
  }
}

function pickPlaneSeries(series: SeriesMeta[]) {
  const out: Record<string, string> = {};
  for (const plane of ["Sagittal", "Coronal", "Axial"]) {
    const cands = series.filter((s) => s.plane === plane && s.include);
    // fat-suppressed fluid-sensitive series first (where most findings show), then the one with the most slices
    cands.sort((a, b) => Number(b.fatSuppression) - Number(a.fatSuppression) || b.slices - a.slices);
    if (cands[0]) out[plane] = cands[0].uid;
  }
  return out;
}

async function adopt(m: Manifest) {
  const geos: Record<string, Geo> = {};
  for (const s of m.series) geos[s.uid] = geometry(s);
  const planeSeries = pickPlaneSeries(m.series);
  const slice: Record<string, number> = {};
  for (const s of m.series) slice[s.uid] = Math.floor(s.slices / 2);
  const active = planeSeries.Sagittal ?? m.series[0]?.uid ?? null;
  setState({
    manifest: m,
    geos,
    volumes: {},
    inputs: {},
    explain: {},
    planeSeries,
    slice,
    active,
    point: null,
    finding: null,
    run: { ...initial.run },
    cursor: -1,
    follow: true,
    error: null,
    loading: { phase: "volumes", progress: 0, label: "Reading volumes" },
  });
  // the shown series first
  const order = [...new Set([active, ...Object.values(planeSeries), ...m.series.map((s) => s.uid)].filter(Boolean) as string[])];
  let n = 0;
  for (const uid of order) {
    const v = await api.volume(m.sid, uid);
    if (getState().manifest?.sid !== m.sid) return;
    setState((s) => ({ volumes: { ...s.volumes, [uid]: v }, loading: { ...s.loading, progress: ++n / order.length } }));
  }
  setState({ loading: { phase: "idle", progress: 1, label: "" } });
}

export async function loadSample(id: string) {
  await clearStudy();
  setState({ loading: { phase: "parsing", progress: 0, label: "Reading the study from this machine" }, error: null });
  try {
    await adopt(await api.loadSample(id));
  } catch (e: any) {
    setState({ error: String(e?.message ?? e), loading: { ...initial.loading } });
  }
}

export async function uploadFiles(files: { path: string; file: File }[]) {
  await clearStudy();
  setState({ loading: { phase: "reading", progress: 0, label: `Packing ${files.length} files` }, error: null });
  try {
    const m = await api.upload(files, (p) =>
      setState({ loading: { phase: p < 0.5 ? "reading" : p < 1 ? "uploading" : "parsing", progress: p, label: p < 0.5 ? "Packing files" : p < 1 ? "Sending to 127.0.0.1" : "Parsing DICOM" } }),
    );
    await adopt(m);
  } catch (e: any) {
    setState({ error: String(e?.message ?? e), loading: { ...initial.loading } });
  }
}

export async function clearStudy() {
  const sid = getState().manifest?.sid;
  stopPlay();
  setState({ manifest: null, volumes: {}, inputs: {}, explain: {}, geos: {}, run: { ...initial.run }, cursor: -1, finding: null, point: null });
  if (sid) await api.purge(sid).catch(() => undefined);
}

export async function runInference() {
  const m = getState().manifest;
  if (!m || getState().run.state === "running") return;
  stopPlay();
  setState({ run: { state: "running", events: [], summary: null, error: null, startedAt: performance.now() }, cursor: -1, follow: true, finding: null, explain: {}, inputs: {} });
  try {
    await api.infer(m.sid, (ev) => {
      if (ev.kind === "result") {
        setState((s) => ({ run: { ...s.run, summary: ev.summary } }));
        return;
      }
      if (ev.kind === "error" && !ev.stage) {
        setState((s) => ({ run: { ...s.run, state: "error", error: ev.error } }));
        return;
      }
      setState((s) => {
        const events = [...s.run.events, ev];
        return { run: { ...s.run, events }, cursor: s.follow ? events.length - 1 : s.cursor };
      });
    });
    const sum = getState().run.summary;
    if (!sum) throw new Error(getState().run.error ?? "the run ended without a result");
    // explanation maps and the classifier's input crops, per series
    const uids = Object.keys(sum.series).filter((u) => sum.series[u].models);
    const ex: Record<string, Explain> = {};
    const inp: Record<string, Vol> = {};
    await Promise.all(
      uids.map(async (u) => {
        ex[u] = await api.explain(m.sid, u);
        inp[u] = await api.modelInput(m.sid, u);
      }),
    );
    // open on the finding with the largest probability
    const best = sum.probs.map((p, i) => [p, i] as const).sort((a, b) => b[0] - a[0])[0]?.[1] ?? null;
    setState((s) => ({ run: { ...s.run, state: "done" }, explain: ex, inputs: inp }));
    if (best !== null) selectFinding(best);
  } catch (e: any) {
    setState((s) => ({ run: { ...s.run, state: "error", error: String(e?.message ?? e) } }));
  }
}

// ------------------------------------------------------------------------------------------- replay
let playTimer: number | undefined;
export function stopPlay() {
  clearTimeout(playTimer);
  setState({ playing: false });
}
export function setCursor(i: number) {
  const n = getState().run.events.length;
  stopPlay();
  setState({ cursor: Math.max(0, Math.min(n - 1, i)), follow: i >= n - 1 });
}
/** Replays the recorded trace with its real relative timing, stretched so short GPU steps stay visible. */
export function play(speed = 1) {
  const { events } = getState().run;
  if (!events.length) return;
  let i = getState().cursor >= events.length - 1 ? 0 : getState().cursor + 1;
  setState({ playing: true, follow: false, cursor: i });
  const tick = () => {
    if (!getState().playing) return;
    if (i >= events.length - 1) {
      setState({ playing: false, follow: true });
      return;
    }
    const dt = Math.max(260, Math.min(1400, (events[i + 1].t - events[i].t) * 0.6)) / speed;
    playTimer = window.setTimeout(() => {
      i++;
      setState({ cursor: i });
      tick();
    }, dt);
  };
  tick();
}

// ------------------------------------------------------------------------------------------- selection
export function selectFinding(f: number | null) {
  const s = getState();
  if (f === null) return setState({ finding: null });
  const sum = s.run.summary;
  if (!sum) return setState({ finding: f });
  // jump every series to the slice where this finding's positive evidence peaks; activate the strongest series
  const slice = { ...s.slice };
  // the series shown in 3D: among those within 0.03 of the strongest for this finding, prefer fat-suppressed, thin-slice
  // series (most anatomy to show the evidence on); the user can open any other from the series list
  let best: { uid: string; p: number } | null = null;
  const maxP = Math.max(...Object.values(sum.series).map((r) => r.probs?.[f] ?? 0));
  const score = (uid: string, p: number) => {
    const m = s.manifest?.series.find((x) => x.uid === uid);
    return p >= maxP - 0.03 ? 1 + Number(!!m?.fatSuppression) * 0.5 + (m?.slices ?? 0) / 1000 : p;
  };
  for (const [uid, r] of Object.entries(sum.series)) {
    const ex = s.explain[uid];
    if (!r.probs || !r.slicePos) continue;
    if (!best || score(uid, r.probs[f]) > score(best.uid, best.p)) best = { uid, p: r.probs[f] };
    if (ex) {
      const { pos } = sliceProfile(ex, f);
      let k = 0;
      for (let i = 1; i < pos.length; i++) if (pos[i] > pos[k]) k = i;
      slice[uid] = Math.round(r.slicePos[k]);
    }
  }
  const patch: Partial<State> = { finding: f, slice };
  if (best) {
    patch.active = best.uid;
    const g = s.geos[best.uid];
    const box = sum.series[best.uid].boxes[f];
    const m = s.manifest!.series.find((x) => x.uid === best!.uid)!;
    const plane = m.plane;
    if (s.planeSeries[plane] !== best.uid) patch.planeSeries = { ...s.planeSeries, [plane]: best.uid };
    // crosshair at the evidence peak inside the box
    const ex = s.explain[best.uid];
    if (g && ex) {
      const r = sum.series[best.uid];
      const { pos } = sliceProfile(ex, f);
      let k = 0;
      for (let i = 1; i < pos.length; i++) if (pos[i] > pos[k]) k = i;
      const cells = ex.h * ex.w;
      const base = (k * ex.k + f) * cells;
      let c = 0;
      for (let i = 1; i < cells; i++) if (ex.contrib[base + i] > ex.contrib[base + c]) c = i;
      const [x0, y0, x1, y1] = r.crop!;
      const u = x0 + ((c % ex.w) + 0.5) / ex.w * (x1 - x0);
      const v = y0 + (Math.floor(c / ex.w) + 0.5) / ex.h * (y1 - y0);
      const w = (r.slicePos![k] + 0.5) / g.depth;
      patch.point = toPatientPoint(g, u, v, w);
    } else if (g && box) {
      patch.point = toPatientPoint(g, (box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2);
    }
  }
  setState(patch);
  if (patch.point) setPoint(patch.point, best?.uid);
}

import { toPatient } from "./geometry";
function toPatientPoint(g: Geo, u: number, v: number, w: number): V3 {
  return toPatient(g, u, v, w);
}

/** Crosshair: every other series jumps to the slice nearest the point. */
export function setPoint(p: V3, from?: string) {
  const s = getState();
  const slice = { ...s.slice };
  for (const [uid, g] of Object.entries(s.geos)) {
    if (uid === from) continue;
    const [, , w] = toUnit(g, p);
    if (w >= -0.05 && w <= 1.05) slice[uid] = sliceOfW(g, w);
  }
  setState({ point: p, slice });
}

export function setSlice(uid: string, k: number) {
  const g = getState().geos[uid];
  if (!g) return;
  setState((s) => ({ slice: { ...s.slice, [uid]: Math.max(0, Math.min(g.depth - 1, k)) } }));
}

export async function editSeries(uid: string, body: Partial<{ fatSuppression: boolean; include: boolean }>) {
  const m = getState().manifest;
  if (!m) return;
  const meta = await api.editSeries(m.sid, uid, body);
  setState((s) => ({
    manifest: s.manifest ? { ...s.manifest, series: s.manifest.series.map((x) => (x.uid === uid ? meta : x)) } : null,
    run: { ...initial.run },
    cursor: -1,
    explain: {},
    inputs: {},
    finding: null,
  }));
}
