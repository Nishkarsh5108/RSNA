// Client for the local inference server (127.0.0.1). Every request goes to this machine; nothing is sent elsewhere.
import { zipSync } from "fflate";

export interface SeriesMeta {
  uid: string;
  plane: "Sagittal" | "Coronal" | "Axial";
  fatSuppression: boolean;
  fsReason: string;
  description: string;
  slices: number;
  rows: number;
  cols: number;
  pixelSpacing: [number, number]; // row spacing, column spacing (mm)
  sliceSpacing: number;
  iop: number[]; // row direction (3) + column direction (3), LPS
  positions: number[][]; // ImagePositionPatient per slice, sorted along the normal
  display: [number, number, number] | null; // display volume shape D, H, W
  window: [number, number];
  tr: number | null;
  te: number | null;
  decompressed: boolean;
  include: boolean;
  note: string;
}

export interface Manifest {
  sid: string;
  studyUid: string;
  shortId: string;
  series: SeriesMeta[];
  skipped: { uid: string; description?: string; slices?: number; reason: string }[];
  source: string;
  reference: { labels: number[] | null; note: string } | null;
  stats: { files: number; nonImage: number; seconds: number; slices: number };
}

export interface ModelInfo {
  id: string;
  params: number;
  labels: string;
  fold: number | null;
  note: string;
}

export interface Status {
  ready: boolean;
  error?: string | null;
  host: string;
  device?: string;
  gpu?: { name: string; totalGB: number; freeGB: number; allocatedGB: number; capability: string } | null;
  precision?: string;
  torch?: string;
  weightsMB?: number;
  loadSeconds?: number;
  busy?: boolean;
  models?: {
    side: { name: string; params: number; input: string };
    detector: { name: string; params: number; input: string };
    classifiers: ModelInfo[];
    classifierInput: string;
  };
  labels?: string[];
}

export interface Sample {
  id: string;
  label: string;
  kind: "unlabelled" | "reference";
  series: number;
  uid?: string;
}

export type Box = number[]; // x0, y0, z0, x1, y1, z1 normalized (native columns, rows, slices)

export interface PipelineEvent {
  stage: "pipeline" | "side" | "detector" | "classifier" | "ensemble";
  kind: string;
  t: number; // ms since the run started (GPU-synchronized)
  vramGB?: number;
  series?: string | string[];
  model?: string;
  models?: string[];
  pRight?: number | null;
  knee?: "left" | "right";
  byDefault?: boolean;
  boxes?: Box[];
  direct?: Box[];
  mirrored?: Box[];
  depth?: number;
  bucket?: number;
  native?: number;
  mirror?: boolean;
  crop?: number[];
  probs?: number[];
  logits?: number[];
  slices?: number;
  cells?: number[];
  channels?: number;
  attributionError?: number;
  ms?: number;
  error?: string;
  fallback?: string[];
  perModel?: Record<string, number[]>;
  seconds?: number;
  peakVramGB?: number;
  maxAttributionError?: number;
  input?: number[];
  framePx?: number;
  cropPx?: number;
}

export interface SeriesResult {
  boxes: Box[];
  detector: { direct?: Box[]; mirrored?: Box[]; depth?: number; bucket?: number; meta?: number[] };
  fallbackBox: boolean;
  probs?: number[];
  crop?: [number, number, number, number]; // x0, y0, x1, y1 in native normalized coordinates
  slicePos?: number[]; // native slice index of each classifier slice
  zRange?: ([number, number] | null)[];
  mirror?: boolean;
  depth?: number;
  native?: number;
  models?: Record<string, { logits: number[]; probs: number[]; attributionError: number }>;
}

export interface Summary {
  study: string;
  knee: "left" | "right";
  pRight: number[];
  failures: { stage: string; series: string; error: string }[];
  models: string[];
  labels: string[];
  probs: number[];
  perModel: Record<string, number[]>;
  spread: { min: number[]; max: number[] } | null;
  series: Record<string, SeriesResult>;
  seconds: number;
  peakVramGB: number | null;
  maxAttributionError: number | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => fetch("/api/status").then((r) => json<Status>(r)),
  samples: () => fetch("/api/samples").then((r) => json<Sample[]>(r)),
  loadSample: (id: string) => fetch(`/api/samples/${id}`, { method: "POST" }).then((r) => json<Manifest>(r)),
  purge: (sid: string) => fetch(`/api/studies/${sid}`, { method: "DELETE" }).then((r) => json<{ purged: boolean }>(r)),
  editSeries: (sid: string, uid: string, body: Partial<{ fatSuppression: boolean; include: boolean }>) =>
    fetch(`/api/studies/${sid}/series/${uid}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<SeriesMeta>(r)),

  /** Upload: dropped files are packed into one uncompressed zip in the browser and posted to the local server. */
  async upload(files: { path: string; file: File }[], onProgress?: (f: number) => void): Promise<Manifest> {
    let body: Blob;
    if (files.length === 1 && /\.zip$/i.test(files[0].path)) {
      body = files[0].file;
    } else {
      const entries: Record<string, [Uint8Array, { level: 0 }]> = {};
      let done = 0;
      for (const { path, file } of files) {
        // files from different folders can share a name (0001.dcm): keep every one under a unique path
        let key = path.replace(/^\/+/, "") || file.name;
        if (entries[key]) key = `${done}_${key}`;
        entries[key] = [new Uint8Array(await file.arrayBuffer()), { level: 0 }];
        onProgress?.((++done / files.length) * 0.5);
      }
      body = new Blob([zipSync(entries)]);
    }
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/studies");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(0.5 + (e.loaded / e.total) * 0.5);
      xhr.onload = () => {
        try {
          const data = JSON.parse(xhr.responseText);
          if (xhr.status >= 400) reject(new Error(data?.detail ?? `${xhr.status}`));
          else resolve(data as Manifest);
        } catch (e) {
          reject(e);
        }
      };
      xhr.onerror = () => reject(new Error("the local server did not answer"));
      xhr.send(body);
    });
  },

  async packed(url: string): Promise<{ header: any; buffer: ArrayBuffer; offset: number }> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${url}`);
    const buffer = await res.arrayBuffer();
    const n = new DataView(buffer).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
    return { header, buffer, offset: 4 + n };
  },

  async volume(sid: string, uid: string) {
    const { header, buffer, offset } = await api.packed(`/api/studies/${sid}/series/${uid}/volume`);
    const [d, h, w] = header.shape as number[];
    return { data: new Uint8Array(buffer, offset, d * h * w), d, h, w };
  },

  async modelInput(sid: string, uid: string) {
    const { header, buffer, offset } = await api.packed(`/api/studies/${sid}/series/${uid}/model-input`);
    const [d, h, w] = header.shape as number[];
    return { data: new Uint8Array(buffer, offset, d * h * w), d, h, w };
  },

  async explain(sid: string, uid: string) {
    const { header, buffer, offset } = await api.packed(`/api/studies/${sid}/series/${uid}/explain`);
    const [d, k, h, w] = header.shape as number[];
    const n = d * k * h * w;
    const at = (i: number) => new Float32Array(buffer, offset + i * n * 4, n);
    return { d, k, h, w, share: at(0), contrib: at(1), boxWeight: at(2) };
  },

  /** Runs inference; calls onEvent for every pipeline step as the GPU produces it (newline-delimited JSON). */
  async infer(sid: string, onEvent: (ev: any) => void, signal?: AbortSignal) {
    const res = await fetch(`/api/studies/${sid}/infer`, { method: "POST", signal });
    if (!res.ok || !res.body) {
      await json(res);
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i: number;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  },
};
