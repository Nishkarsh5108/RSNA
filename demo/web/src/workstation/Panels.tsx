import { useMemo, useRef, useState } from "react";
import type { PipelineEvent } from "../lib/api";
import { FINDINGS } from "../lib/findings";
import {
  editSeries, loadSample, peakSlices, pipelineView, play, runInference, selectFinding, setCursor, setOverlay, setSlice, setState, stopPlay, uploadFiles, useStore,
} from "./store";

const PLANE = { Sagittal: "SAG", Coronal: "COR", Axial: "AX" } as const;
const pct = (p: number) => (p * 100).toFixed(1);

// ------------------------------------------------------------------------------------------------ series
export function SeriesPanel() {
  const m = useStore((s) => s.manifest);
  const active = useStore((s) => s.active);
  const finding = useStore((s) => s.finding);
  const sum = useStore((s) => s.run.summary);
  const running = useStore((s) => s.run.state === "running");
  const side = sum?.knee;
  if (!m) return null;
  return (
    <section className="panel">
      <header className="panel-h">
        <span className="label label--fg">Study</span>
        <span className="label num">{m.shortId}</span>
      </header>
      <div className="study-meta mono">
        <div>
          <span className="k">source</span>
          <span>{m.source}</span>
        </div>
        <div>
          <span className="k">images</span>
          <span>
            {m.stats.slices} in {m.series.length} series · parsed {m.stats.seconds.toFixed(1)} s
          </span>
        </div>
        {side && (
          <div>
            <span className="k">knee</span>
            <span className="accent">{side.toUpperCase()}</span>
          </div>
        )}
      </div>
      <div className="series-list">
        {m.series.map((s) => {
          const p = finding !== null ? sum?.series[s.uid]?.probs?.[finding] : undefined;
          return (
            <div key={s.uid} className={`series-row ${active === s.uid ? "on" : ""} ${s.include ? "" : "off"}`} onClick={() => setState({ active: s.uid })}>
              <span className="sr-plane mono">{PLANE[s.plane]}</span>
              <span className="sr-desc" title={s.description}>
                {s.description || s.uid.slice(-8)}
                <span className="sr-sub mono">
                  {s.slices} sl · {s.pixelSpacing[1].toFixed(2)} mm · {s.sliceSpacing.toFixed(1)} mm
                </span>
              </span>
              <button
                className={`sr-fs mono ${s.fatSuppression ? "on" : ""}`}
                title={`Fat suppression (detector input): ${s.fsReason}. Click to change.`}
                disabled={running}
                onClick={(e) => {
                  e.stopPropagation();
                  editSeries(s.uid, { fatSuppression: !s.fatSuppression });
                }}
              >
                FS
              </button>
              {p !== undefined ? (
                <span className="sr-p num">{p.toFixed(2)}</span>
              ) : (
                <input
                  type="checkbox"
                  className="sr-inc"
                  checked={s.include}
                  disabled={running}
                  title="Include this series in inference"
                  onClick={(e) => e.stopPropagation()}
                  onChange={() => editSeries(s.uid, { include: !s.include })}
                />
              )}
            </div>
          );
        })}
      </div>
      {m.skipped.length > 0 && (
        <div className="skipped mono">
          {m.skipped.map((s) => (
            <div key={s.uid}>
              — {s.description || s.uid.slice(-8)}: {s.reason}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------------------------------------ pipeline
const STAGES: { id: PipelineEvent["stage"]; n: string; name: string; net: string }[] = [
  { id: "side", n: "01", name: "Knee side", net: "ResNet-18d · 11 M params" },
  { id: "detector", n: "02", name: "ROI detector", net: "ConvNeXt-T, 3D fusion · 29 M" },
  { id: "classifier", n: "03", name: "Box-pooling classifiers", net: "ConvNeXt-T · 31 M each" },
  { id: "ensemble", n: "04", name: "Ensemble", net: "mean over series, then models" },
];

export function PipelinePanel() {
  const run = useStore((s) => s.run);
  const cursor = useStore((s) => s.cursor);
  const playing = useStore((s) => s.playing);
  const m = useStore((s) => s.manifest);
  const status = useStore((s) => s.status);
  const view = useMemo(() => pipelineView(run.events, cursor), [run.events, cursor]);
  const track = useRef<HTMLDivElement>(null);
  const n = run.events.length;
  const T = n ? run.events[n - 1].t : 1;
  const timing = (id: string) => {
    const evs = run.events.filter((e) => e.stage === id);
    if (!evs.length) return null;
    return evs[evs.length - 1].t - evs[0].t;
  };
  const stageState = (id: string) => {
    const order = ["pipeline", "side", "detector", "classifier", "ensemble"];
    const cur = order.indexOf(view.stage === "ingest" ? "pipeline" : view.stage);
    const me = order.indexOf(id);
    if (!n) return "idle";
    if (me < cur || (id === "ensemble" && view.done)) return "done";
    if (me === cur) return run.state === "running" && cursor === n - 1 ? "live" : "here";
    return "todo";
  };
  const seek = (e: React.PointerEvent) => {
    const r = track.current!.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width) * T;
    let i = 0;
    while (i < n - 1 && run.events[i + 1].t <= t) i++;
    setCursor(i);
  };
  return (
    <section className="panel panel-pipe">
      <header className="panel-h">
        <span className="label label--fg">Inference</span>
        <span className="label">{status?.precision ? "fp16 autocast · local GPU" : ""}</span>
      </header>
      <div className="run-row">
        <button className="btn btn--accent run-btn" disabled={!m || run.state === "running" || !status?.ready} onClick={() => runInference()}>
          {run.state === "running" ? (
            <>
              <span className="dot dot--live" style={{ background: "#000" }} /> Running on GPU
            </>
          ) : run.state === "done" ? (
            <>
              Run again <span className="arrow">↻</span>
            </>
          ) : (
            <>
              Run inference <span className="arrow">→</span>
            </>
          )}
        </button>
      </div>
      {run.error && <div className="err mono">{run.error}</div>}
      <ol className="stages">
        {STAGES.map((st) => {
          const ss = stageState(st.id);
          const ms = timing(st.id);
          return (
            <li key={st.id} className={`stage s-${ss}`}>
              <span className="stage-n num">{st.n}</span>
              <span className="stage-name">
                {st.name}
                <span className="stage-net mono">{st.net}</span>
              </span>
              <span className="stage-t num">{ms !== null && ss !== "todo" ? (ms < 10 ? "<0.01 s" : `${(ms / 1000).toFixed(2)} s`) : ss === "live" ? "…" : ""}</span>
            </li>
          );
        })}
      </ol>
      {n > 0 && (
        <div className="trace">
          <div className="trace-track" ref={track} onPointerDown={(e) => ((e.target as HTMLElement).setPointerCapture(e.pointerId), seek(e))} onPointerMove={(e) => e.buttons && seek(e)}>
            {run.events.map((e, i) => (
              <span key={i} className={`tick t-${e.stage} ${i <= cursor ? "past" : ""} ${e.kind === "error" ? "t-err" : ""}`} style={{ left: `${(e.t / T) * 100}%` }} />
            ))}
            <span className="trace-head" style={{ left: `${((run.events[Math.max(0, cursor)]?.t ?? 0) / T) * 100}%` }} />
          </div>
          <div className="trace-ctl">
            <button className="tool" onClick={() => setCursor(0)} title="First step">
              |◀
            </button>
            <button className="tool" onClick={() => setCursor(cursor - 1)} title="Previous step">
              ◀
            </button>
            <button className="tool on" onClick={() => (playing ? stopPlay() : play(1))} disabled={run.state === "running"} title="Replay the recorded trace">
              {playing ? "❚❚" : "▶"}
            </button>
            <button className="tool" onClick={() => setCursor(cursor + 1)} title="Next step">
              ▶
            </button>
            <button className="tool" onClick={() => setCursor(n - 1)} title="Last step">
              ▶|
            </button>
            <span className="trace-pos num">
              {cursor + 1}/{n} · {((run.events[Math.max(0, cursor)]?.t ?? 0) / 1000).toFixed(2)} s
            </span>
          </div>
          <div className="label trace-note">Replay of the recorded GPU trace; times are real (CUDA-synchronized).</div>
        </div>
      )}
      {run.summary && (
        <div className="facts mono">
          <div>
            <span className="k">total</span>
            <span>{run.summary.seconds.toFixed(2)} s</span>
          </div>
          <div>
            <span className="k">peak VRAM</span>
            <span>
              {run.summary.peakVramGB?.toFixed(2)} / {status?.gpu?.totalGB.toFixed(1)} GB
            </span>
          </div>
          <div>
            <span className="k">attribution</span>
            <span title="Largest |Σ cell contributions + bias − model logit| over all series, models and findings. Zero in fp32; the residual is the fp16 rounding inside the model's own pooling.">
              Σ = logit ± {run.summary.maxAttributionError?.toExponential(1)}
            </span>
          </div>
          {run.summary.failures.length > 0 && (
            <div>
              <span className="k">skipped</span>
              <span>{run.summary.failures.length} series step(s)</span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------------------------------------ findings
export function FindingsPanel() {
  const fullSum = useStore((s) => s.run.summary);
  const finding = useStore((s) => s.finding);
  const overlay = useStore((s) => s.overlay);
  const modelView = useStore((s) => s.modelView);
  const m = useStore((s) => s.manifest);
  const status = useStore((s) => s.status);
  const runState = useStore((s) => s.run.state);
  const events = useStore((s) => s.run.events);
  const cursor = useStore((s) => s.cursor);
  const ref = m?.reference?.labels ?? null;
  // while running or replaying: the mean over the series finished up to the trace position
  const atEnd = runState !== "running" && cursor >= events.length - 1;
  const sum = atEnd ? fullSum : null;
  const live = useMemo(() => {
    if (sum || !events.length) return null;
    const v = pipelineView(events, cursor);
    const ps = Object.values(v.seriesProbs);
    if (!ps.length) return null;
    return FINDINGS.map((_, i) => ps.reduce((a, p) => a + p[i], 0) / ps.length);
  }, [sum, events, cursor]);
  const probs = sum?.probs ?? live;
  const models = status?.models?.classifiers ?? [];
  return (
    <section className="panel panel-find">
      <header className="panel-h">
        <span className="label label--fg">Findings</span>
        <span className="label">{sum ? `${sum.models.length}-model ensemble` : live ? "partial · series done so far" : fullSum ? "replay" : "awaiting inference"}</span>
      </header>
      <div className="find-cols label">
        <span>finding</span>
        <span>probability · training prevalence ▏</span>
        {ref && <span title="Reference label shipped with the dataset">ref</span>}
      </div>
      <div className="find-list">
        {FINDINGS.map((f, i) => {
          const p = probs?.[i];
          const flagged = p !== undefined && p >= 0.5;
          const lo = sum?.spread?.min[i];
          const hi = sum?.spread?.max[i];
          return (
            <button
              key={f.key}
              aria-label={`${f.name}${p !== undefined ? `, ${pct(p)} percent` : ""}`}
              className={`find-row ${finding === i ? "on" : ""} ${flagged ? "flag" : ""}`}
              disabled={!sum}
              onClick={() => selectFinding(finding === i ? null : i)}
            >
              <span className="fr-code mono">{f.short}</span>
              <span className="fr-name">{f.name}</span>
              <span className="fr-p num">{p !== undefined ? pct(p) : "—"}</span>
              <span className="fr-bar">
                {p !== undefined && <span className="fr-fill" style={{ width: `${p * 100}%` }} />}
                {lo !== undefined && hi !== undefined && <span className="fr-spread" style={{ left: `${lo * 100}%`, width: `${Math.max(0.6, (hi - lo) * 100)}%` }} />}
                <span className="fr-prev" style={{ left: `${f.prevalence * 100}%` }} title={`training prevalence ${pct(f.prevalence)}%`} />
              </span>
              {ref && <span className={`fr-ref mono ${ref[i] === 1 ? "pos" : ""}`}>{ref[i] === 1 ? "+" : "·"}</span>}
            </button>
          );
        })}
      </div>
      <div className="find-legend label">
        bar = ensemble mean · whisker = range over models · tick = training prevalence · <span className="accent">■</span> ≥ 50%
      </div>
      {sum && (
        <div className="seg">
          {(["evidence", "attention", "lesion", "box"] as const).map((o) => (
            <button
              key={o}
              className={`tool ${overlay === o ? "on" : ""}`}
              onClick={() => setOverlay(o)}
              title={
                o === "evidence"
                  ? "What drove the score: each cell's exact contribution to the logit"
                  : o === "attention"
                    ? "Where the model pooled: each cell's share of the pooled feature"
                    : o === "lesion"
                      ? "Where the network's lesion head places a lesion (trained on per-slice lesion boxes; not used for the score)"
                      : "The detector's ROI box and the classifier's crop"
              }
            >
              {o === "evidence" ? "Evidence" : o === "attention" ? "Attention" : o === "lesion" ? "Lesion" : "Boxes"}
            </button>
          ))}
          <span className="tool-sep" />
          <button className={`tool ${modelView ? "on" : ""}`} onClick={() => setState({ modelView: !modelView })} title="Show the 256 px crop the classifier actually read">
            Model input
          </button>
        </div>
      )}
      {sum && finding !== null && <FindingDetail i={finding} />}
      {!fullSum && runState !== "running" && m && (
        <p className="body-s find-empty">
          Run inference to read this study. Every step runs on this machine's GPU; you can watch the boxes, crops and feature cells as they are computed, then replay the trace.
        </p>
      )}
      {models.length > 0 && !sum && (
        <div className="models mono">
          {models.map((x) => (
            <div key={x.id}>
              <span className="k">{x.id}</span>
              <span>
                {x.labels}, fold {x.fold}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function FindingDetail({ i }: { i: number }) {
  const sum = useStore((s) => s.run.summary)!;
  const m = useStore((s) => s.manifest)!;
  const status = useStore((s) => s.status);
  const overlay = useStore((s) => s.overlay);
  const active = useStore((s) => s.active);
  const explain = useStore((s) => s.explain);
  const f = FINDINGS[i];
  const models = status?.models?.classifiers ?? [];
  const series = m.series.filter((s) => sum.series[s.uid]?.probs);
  const show = (uid: string) => {
    const plane = m.series.find((x) => x.uid === uid)!.plane;
    setState((st) => ({ active: uid, planeSeries: { ...st.planeSeries, [plane]: uid } }));
  };
  // where the shown map is concentrated in the active series: its top slices, clickable
  const act = active ? m.series.find((x) => x.uid === active) : undefined;
  const ex = active ? explain[active] : undefined;
  const r = active ? sum.series[active] : undefined;
  const mode = overlay === "box" ? "evidence" : overlay;
  const peaks = ex && r?.slicePos ? peakSlices(ex, i, mode, 4) : [];
  return (
    <div className="detail">
      <div className="detail-h">
        <span className="h3">{f.name}</span>
        <span className="detail-p num">{pct(sum.probs[i])}%</span>
      </div>
      <div className="detail-region label">{f.region}</div>
      <div className="detail-block">
        <div className="label">per model (series mean)</div>
        {sum.models.map((k) => {
          const info = models.find((x) => x.id === k);
          const p = sum.perModel[k][i];
          return (
            <div key={k} className="mini-row">
              <span className="mono mr-k">{k}</span>
              <span className="mini-bar">
                <span style={{ width: `${p * 100}%` }} />
              </span>
              <span className="num mr-v">{pct(p)}</span>
              <span className="mr-note label">{info?.labels}</span>
            </div>
          );
        })}
      </div>
      <div className="detail-block">
        <div className="label">per series (model mean) — the study is their mean</div>
        {series.map((s) => {
          const p = sum.series[s.uid].probs![i];
          return (
            <button key={s.uid} className={`mini-row as-btn ${active === s.uid ? "on" : ""}`} onClick={() => show(s.uid)}>
              <span className="mono mr-k">{PLANE[s.plane]}</span>
              <span className="mini-bar">
                <span style={{ width: `${p * 100}%` }} />
              </span>
              <span className="num mr-v">{pct(p)}</span>
              <span className="mr-note label">{(s.description || "").slice(0, 18)}</span>
            </button>
          );
        })}
      </div>
      {act && peaks.length > 0 && (
        <div className="detail-block">
          <div className="label">
            {mode === "evidence" ? "evidence" : mode === "attention" ? "pooling weight" : "lesion head"} by slice · {PLANE[act.plane]} {(act.description || "").slice(0, 16)}
          </div>
          {peaks.map((p) => {
            const native = Math.round(r!.slicePos![p.k]);
            return (
              <button key={p.k} className="mini-row as-btn" onClick={() => (show(act.uid), setSlice(act.uid, native))}>
                <span className="mono mr-k">Im {native + 1}</span>
                <span className="mini-bar">
                  <span style={{ width: `${p.frac * 100}%` }} />
                </span>
                <span className="num mr-v">{Math.round(p.frac * 100)}%</span>
                <span className="mr-note label">of {act.slices}</span>
              </button>
            );
          })}
        </div>
      )}
      <p className="detail-note body-s">
        {overlay === "evidence"
          ? "Yellow cells: each feature cell's exact contribution to this finding's logit (pooling share × head · feature), averaged over the models. It shows what drove the score, which need not be the most visible lesion: the model reads 14 mm cells inside the finding's box and can lean on anatomy around a lesion. Compare with Lesion."
          : overlay === "attention"
            ? "White cells: where the model pooled — each cell's share of the finding's pooled feature (ROI box × learned in-box attention). It shows where the model looked, not what it found there."
            : overlay === "lesion"
              ? "Outlined cells: the network's lesion head — trained only during training, on per-slice lesion boxes drawn for each finding — gives the probability that each 14 mm cell is covered by a lesion. It reads the same features but does not feed the probability above, so it is an independent localisation."
              : "Boxes: the detector's 3D box per finding (yellow) and the 112 mm window the classifier reads (dashed)."}
      </p>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ intake
async function walk(entry: any, path: string, out: { path: string; file: File }[]) {
  if (entry.isFile) {
    const file: File = await new Promise((res, rej) => entry.file(res, rej));
    out.push({ path: path + file.name, file });
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    for (;;) {
      const batch: any[] = await new Promise((res, rej) => reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const e of batch) await walk(e, path + entry.name + "/", out);
    }
  }
}

export function DropZone() {
  const samples = useStore((s) => s.samples);
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);
  const status = useStore((s) => s.status);
  const [over, setOver] = useState(false);
  const dirInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const busy = loading.phase !== "idle";
  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const items = Array.from(e.dataTransfer.items);
    const out: { path: string; file: File }[] = [];
    for (const it of items) {
      const entry = (it as any).webkitGetAsEntry?.();
      if (entry) await walk(entry, "", out);
      else if (it.kind === "file") out.push({ path: it.getAsFile()!.name, file: it.getAsFile()! });
    }
    if (out.length) uploadFiles(out);
  };
  const fromInput = (list: FileList | null) => {
    if (!list?.length) return;
    uploadFiles(Array.from(list).map((f) => ({ path: (f as any).webkitRelativePath || f.name, file: f })));
  };
  return (
    <div
      className={`drop ${over ? "over" : ""} ${busy ? "busy" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <div className="drop-frame crop">
        <div className="drop-main">
          <div className="label">Intake</div>
          <h2 className="drop-title">
            Drop a knee
            <br />
            MRI study
          </h2>
          <p className="body-s drop-sub">A DICOM folder, a set of .dcm files or a .zip. One study; sagittal, coronal and axial MR series with patient geometry. Localizers are skipped automatically.</p>
          <div className="drop-btns">
            <button className="btn" disabled={busy} onClick={() => dirInput.current?.click()}>
              Choose folder
            </button>
            <button className="btn" disabled={busy} onClick={() => fileInput.current?.click()}>
              Choose files
            </button>
          </div>
          <input ref={dirInput} type="file" hidden {...({ webkitdirectory: "", directory: "" } as any)} multiple onChange={(e) => fromInput(e.target.files)} />
          <input ref={fileInput} type="file" hidden multiple accept=".dcm,.zip,application/dicom,application/zip" onChange={(e) => fromInput(e.target.files)} />
          <div className="privacy mono">
            <span className="dot dot--accent" /> Files go only to the local server at {status?.host ?? "127.0.0.1:8000"} — parsed, cached and run on this machine, deleted on purge. Nothing leaves it.
          </div>
          {busy && (
            <div className="progress">
              <div className="label">{loading.label}…</div>
              <div className="progress-track">
                <span style={{ width: `${Math.max(4, loading.progress * 100)}%` }} />
              </div>
            </div>
          )}
          {error && <div className="err mono">{error}</div>}
        </div>
        {samples.length > 0 && (
          <div className="samples">
            <div className="label">Or open a local sample</div>
            <div className="samples-note body-s">Read from this machine's copy of the RSNA dataset; never bundled with the site.</div>
            {samples.map((s) => (
              <button key={s.id} className="sample" disabled={busy} onClick={() => loadSample(s.id)}>
                <span className="sample-l">
                  {s.label}
                  <span className="sample-k label">{s.kind === "reference" ? "held-out · reference labels" : "public test · unlabelled"}</span>
                </span>
                <span className="sample-n num">{s.series} series</span>
                <span className="arrow">→</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
