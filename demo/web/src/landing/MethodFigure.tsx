// The method figure: one SVG whose layers change with the active step. Geometry follows the real pipeline (crop rule,
// 8 x 8 cells, box share of each cell's 4 x 4 sub-points, mean over series then models); the anatomy and all numbers
// shown here are illustrative. The workstation shows the model's real boxes, weights and contributions.
import { useMemo } from "react";
import { FINDINGS } from "../lib/findings";
import { AX, COR, HOTSPOT, SAG, SCHEMATIC_BOX, cropOf, hash, makeIso, planeMatrix, pts, type V3 } from "./anatomy";

const W = 960;
const H = 640;

function Sag({ faint = false }: { faint?: boolean }) {
  return (
    <g className={faint ? "art art--faint" : "art"}>
      <path className="art-soft" d={SAG.skinA} />
      <path className="art-soft" d={SAG.skinP} />
      <path className="art-bone" d={SAG.femur} />
      <path className="art-bone" d={SAG.tibia} />
      <path className="art-bone" d={SAG.patella} />
      <path className="art-men" d={SAG.menA} />
      <path className="art-men" d={SAG.menP} />
      <path className="art-lig" d={SAG.acl} />
      <path className="art-tendon" d={SAG.ptendon} />
      <path className="art-tendon" d={SAG.qtendon} />
    </g>
  );
}

function Cor() {
  return (
    <g className="art">
      <path className="art-soft" d={COR.skin} />
      <path className="art-bone" d={COR.femur} />
      <path className="art-bone" d={COR.tibia} />
      <path className="art-bone" d={COR.fibula} />
      <path className="art-men" d={COR.menL} />
      <path className="art-men" d={COR.menM} />
      <path className="art-tendon" d={COR.mcl} />
      <path className="art-tendon" d={COR.lcl} />
    </g>
  );
}

function Ax() {
  return (
    <g className="art">
      <path className="art-soft" d={AX.skin} />
      <path className="art-bone" d={AX.femur} />
      <path className="art-bone" d={AX.patella} />
    </g>
  );
}

function Tag({ x, y, children, anchor = "start", accent = false }: { x: number; y: number; children: React.ReactNode; anchor?: "start" | "middle" | "end"; accent?: boolean }) {
  return (
    <text x={x} y={y} className={accent ? "fig-tag fig-tag--accent" : "fig-tag"} textAnchor={anchor}>
      {children}
    </text>
  );
}

// ---------------------------------------------------------------- 01 study
function FigStudy({ progress }: { progress: number }) {
  const N = 11;
  const spread = 0.42 + 0.22 * progress;
  const stacks = [
    { cx: 150, label: "SAGITTAL", seq: "PD FS · 40 slices · 3.0 mm", kind: "sag" as const },
    { cx: 475, label: "CORONAL", seq: "T1 · 30 slices · 3.0 mm", kind: "cor" as const },
    { cx: 800, label: "AXIAL", seq: "PD FS · 42 slices · 3.0 mm", kind: "ax" as const },
  ];
  return (
    <g>
      {stacks.map((st, si) => {
        const iso = makeIso(st.cx, st.kind === "ax" ? 320 : 250, 124);
        const slices = Array.from({ length: N }, (_, i) => (i / (N - 1)) * spread);
        return (
          <g key={st.kind}>
            {slices.map((p, i) => {
              let corners: V3[];
              let m = "";
              if (st.kind === "sag") {
                corners = [[p, 1, 1], [p, 0, 1], [p, 0, 0], [p, 1, 0]];
                m = planeMatrix(iso, [p, 1, 1], [0, -1, 0], [0, 0, -1]);
              } else if (st.kind === "cor") {
                corners = [[0, p, 1], [1, p, 1], [1, p, 0], [0, p, 0]];
                m = planeMatrix(iso, [0, p, 1], [1, 0, 0], [0, 0, -1]);
              } else {
                corners = [[0, 0, p], [1, 0, p], [1, 1, p], [0, 1, p]];
                m = planeMatrix(iso, [0, 0, p], [1, 0, 0], [0, 1, 0]);
              }
              const front = i === N - 1;
              return (
                <g key={i} className="in" style={{ ["--i" as string]: si * 4 + i }}>
                  <polygon points={pts(corners.map(iso))} className={front ? "slice slice--front" : "slice"} />
                  {front && <g transform={m}>{st.kind === "sag" ? <Sag /> : st.kind === "cor" ? <Cor /> : <Ax />}</g>}
                </g>
              );
            })}
            <g className="in" style={{ ["--i" as string]: 14 + si }}>
              <Tag x={st.cx - 92} y={520}>
                {st.label}
              </Tag>
              <Tag x={st.cx - 92} y={538}>
                <tspan className="fig-dim">{st.seq}</tspan>
              </Tag>
            </g>
          </g>
        );
      })}
      <line x1={40} x2={920} y1={566} y2={566} className="fig-rule" />
      <Tag x={40} y={592}>
        ONE STUDY = SEVERAL SERIES · EACH SORTED ALONG ITS SLICE NORMAL n = r × c
      </Tag>
    </g>
  );
}

// ---------------------------------------------------------------- 02 knee side
function FigSide({ on }: { on: boolean }) {
  const rows = [
    ["COR PD FS", 0.03],
    ["COR T1", 0.05],
    ["AX PD FS", 0.02],
  ] as const;
  const mean = rows.reduce((a, r) => a + r[1], 0) / rows.length;
  return (
    <g>
      <g transform="translate(30 130) scale(1.05)">
        <rect width={256} height={256} className="frame" />
        <g transform="translate(256 0) scale(-1 1)">
          <Cor />
        </g>
        <Tag x={0} y={-14}>A · AS ACQUIRED</Tag>
        <Tag x={0} y={278}><tspan className="fig-dim">fibula on the viewer's right → a left knee</tspan></Tag>
        <Tag x={4} y={130} accent>R</Tag>
        <Tag x={244} y={130} accent>L</Tag>
      </g>
      <g className="in" style={{ ["--i" as string]: 3 }}>
        <line x1={312} x2={376} y1={264} y2={264} className="fig-arrow" markerEnd="url(#arrow)" />
        <Tag x={344} y={250} anchor="middle">x → −x</Tag>
      </g>
      <g transform="translate(398 130) scale(1.05)">
        <rect width={256} height={256} className="frame frame--accent" />
        <g className={on ? "mirror mirror--done" : "mirror"}>
          <Cor />
        </g>
        <Tag x={0} y={-14}>B · AS THE MODELS SEE IT</Tag>
        <Tag x={0} y={278}><tspan className="fig-dim">every knee read as a right knee</tspan></Tag>
      </g>
      <g transform="translate(712 150)">
        <Tag x={0} y={0}>RESNET-18D · p(RIGHT)</Tag>
        {rows.map(([name, p], i) => (
          <g key={name} transform={`translate(0 ${34 + i * 40})`}>
            <g className="in" style={{ ["--i" as string]: 4 + i }}>
              <Tag x={0} y={0}><tspan className="fig-dim">{name}</tspan></Tag>
              <line x1={0} x2={200} y1={12} y2={12} className="fig-rule" />
              <line x1={100} x2={100} y1={6} y2={18} className="fig-rule" />
              <circle cx={200 * p} cy={12} r={4} className="fig-dot" />
              <Tag x={200} y={0} anchor="end">{p.toFixed(2)}</Tag>
            </g>
          </g>
        ))}
        <g transform="translate(0 176)"><g className="in" style={{ ["--i" as string]: 8 }}>
          <line x1={0} x2={200} y1={0} y2={0} className="fig-rule" />
          <Tag x={0} y={24}>MEAN {mean.toFixed(2)} &lt; 0.5</Tag>
          <Tag x={0} y={44} accent>→ LEFT · MIRROR COR + AX</Tag>
          <Tag x={0} y={64}><tspan className="fig-dim">sagittal series stay as acquired</tspan></Tag>
          <Tag x={0} y={98}><tspan className="fig-dim">illustrative values</tspan></Tag>
        </g></g>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------- 03 boxes
function Cuboid({ iso, b, depth, cls }: { iso: (p: V3) => [number, number]; b: number[]; depth: number; cls: string }) {
  const [px0, py0, px1, py1, s0, s1] = b;
  const x = [s0 * depth, s1 * depth];
  const y = [1 - px1 / 256, 1 - px0 / 256];
  const z = [1 - py1 / 256, 1 - py0 / 256];
  const c = (i: number, j: number, k: number) => iso([x[i], y[j], z[k]]);
  const edges: [number[], number[]][] = [
    [[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]], [[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]],
    [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]], [[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]],
  ];
  const face = [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)];
  return (
    <g className={cls}>
      <polygon points={pts(face)} className="cuboid-face" />
      {edges.map(([a, bb], i) => {
        const p = c(a[0], a[1], a[2]);
        const q = c(bb[0], bb[1], bb[2]);
        return <line key={i} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} />;
      })}
    </g>
  );
}

function Bumps({ from, to, at, normal, label }: { from: [number, number]; to: [number, number]; at: [number, number]; normal: [number, number]; label: string }) {
  // two boundary distributions (softmax over bins) along one axis; the box edge is each one's expectation
  const path = (t: number) => {
    const out: string[] = [];
    for (let i = 0; i <= 40; i++) {
      const s = i / 40;
      const g = Math.exp(-((s - t) ** 2) / (2 * 0.035 ** 2));
      const x = from[0] + (to[0] - from[0]) * s + normal[0] * g * 26;
      const y = from[1] + (to[1] - from[1]) * s + normal[1] * g * 26;
      out.push(`${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`);
    }
    return out.join(" ");
  };
  return (
    <g className="bumps">
      <line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} className="fig-rule" />
      <path d={path(at[0])} />
      <path d={path(at[1])} />
      <text x={to[0] + normal[0] * 8 + 6} y={to[1] + normal[1] * 8 + 4} className="fig-tag fig-dim">{label}</text>
    </g>
  );
}

function FigBoxes({ finding }: { finding: string }) {
  const iso = makeIso(300, 250, 230);
  const D = 0.62;
  const N = 14;
  const slices = Array.from({ length: N }, (_, i) => (i / (N - 1)) * D);
  const b = SCHEMATIC_BOX[finding];
  const lift = 34;
  return (
    <g>
      {slices.map((p, i) => (
        <polygon key={i} points={pts(([[p, 1, 1], [p, 0, 1], [p, 0, 0], [p, 1, 0]] as V3[]).map(iso))} className="slice slice--quiet in" style={{ ["--i" as string]: i }} />
      ))}
      <g transform={planeMatrix(iso, [D, 1, 1], [0, -1, 0], [0, 0, -1])}>
        <Sag faint />
      </g>
      {FINDINGS.filter((f) => f.key !== finding).map((f) => (
        <Cuboid key={f.key} iso={iso} b={SCHEMATIC_BOX[f.key]} depth={D} cls="cuboid" />
      ))}
      <Cuboid iso={iso} b={b} depth={D} cls="cuboid cuboid--on" />
      {/* the six boundary distributions of the selected box */}
      <Bumps
        from={iso([D, 1, 0]).map((v, i) => v + [0, lift][i]) as [number, number]}
        to={iso([D, 0, 0]).map((v, i) => v + [0, lift][i]) as [number, number]}
        at={[b[0] / 256, b[2] / 256]}
        normal={[0, 1]}
        label="A–P"
      />
      <Bumps
        from={iso([D, 0, 1]).map((v, i) => v + [lift, 0][i]) as [number, number]}
        to={iso([D, 0, 0]).map((v, i) => v + [lift, 0][i]) as [number, number]}
        at={[b[1] / 256, b[3] / 256]}
        normal={[1, 0]}
        label="S–I"
      />
      <Bumps
        from={iso([0, 1, 0]).map((v, i) => v + [-lift, lift * 0.6][i]) as [number, number]}
        to={iso([D, 1, 0]).map((v, i) => v + [-lift, lift * 0.6][i]) as [number, number]}
        at={[b[4], b[5]]}
        normal={[-0.5, 0.86]}
        label="SLICE"
      />
      <g transform="translate(700 70)">
        <Tag x={0} y={0}>ROI DETECTOR · CONVNEXT-T</Tag>
        <Tag x={0} y={20}><tspan className="fig-dim">160 × 160 · native depth ≤ 192</tspan></Tag>
        <Tag x={0} y={40}><tspan className="fig-dim">4 stages fused top-down in 3D</tspan></Tag>
        <Tag x={0} y={60}><tspan className="fig-dim">FiLM: plane · side · fat-sat · FOV</tspan></Tag>
        <line x1={0} x2={220} y1={80} y2={80} className="fig-rule" />
        <Tag x={0} y={104} accent>{FINDINGS.find((f) => f.key === finding)?.name.toUpperCase()}</Tag>
        <Tag x={0} y={124}><tspan className="fig-dim">6 boundaries, each a distribution</tspan></Tag>
        <Tag x={0} y={144}><tspan className="fig-dim">over bins; edge = its expectation</tspan></Tag>
        <Tag x={0} y={176}><tspan className="fig-dim">+ mirrored pass, averaged (TTA)</tspan></Tag>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------- 04 crop & encode
const CROP = cropOf();

function CropArt({ x, y, size, id }: { x: number; y: number; size: number; id: string }) {
  const [c0, r0, c1] = CROP;
  const k = size / (c1 - c0);
  return (
    <g>
      <clipPath id={id}>
        <rect x={x} y={y} width={size} height={size} />
      </clipPath>
      <g clipPath={`url(#${id})`}>
        <g transform={`translate(${x - c0 * k} ${y - r0 * k}) scale(${k})`}>
          <Sag />
        </g>
      </g>
      <rect x={x} y={y} width={size} height={size} className="frame frame--accent" />
    </g>
  );
}

function Grid({ x, y, size, n = 8, cls = "grid-line" }: { x: number; y: number; size: number; n?: number; cls?: string }) {
  const lines = [];
  for (let i = 1; i < n; i++) {
    const t = (i / n) * size;
    lines.push(<line key={`v${i}`} x1={x + t} x2={x + t} y1={y} y2={y + size} className={cls} style={{ ["--i" as string]: i }} />);
    lines.push(<line key={`h${i}`} x1={x} x2={x + size} y1={y + t} y2={y + t} className={cls} style={{ ["--i" as string]: i }} />);
  }
  return <g>{lines}</g>;
}

function FigCrop({ progress }: { progress: number }) {
  const fx = 50, fy = 130, fs = 300, k = fs / 256;
  const [c0, r0, c1, r1] = CROP;
  const sheets = 4;
  return (
    <g>
      <rect x={fx} y={fy} width={fs} height={fs} className="frame" />
      <g transform={`translate(${fx} ${fy}) scale(${k})`}>
        <Sag />
      </g>
      <rect x={fx + c0 * k} y={fy + r0 * k} width={(c1 - c0) * k} height={(r1 - r0) * k} className="crop-rect" />
      <Tag x={fx} y={fy - 16}>FRAME · 140 MM AT 384 PX</Tag>
      <Tag x={fx + c0 * k} y={fy + fs + 22} accent>CROP · 112 MM AT 256 PX</Tag>
      <Tag x={fx} y={fy + fs + 42}><tspan className="fig-dim">centred on the ACL, meniscus and OA boxes</tspan></Tag>
      <line x1={fx + fs + 18} x2={492} y1={fy + fs / 2} y2={fy + fs / 2} className="fig-arrow" markerEnd="url(#arrow)" />
      {Array.from({ length: sheets }, (_, i) => {
        const o = (sheets - i) * (10 + 10 * progress);
        return <rect key={i} x={520 + o} y={110 - o} width={300} height={300} className="sheet" />;
      })}
      <CropArt x={520} y={110} size={300} id="crop-a" />
      <g className="grid-draw">
        <Grid x={520} y={110} size={300} />
      </g>
      <Tag x={520} y={92}>CONVNEXT-T · ONE SLICE AT A TIME</Tag>
      <Tag x={520} y={438} accent>8 × 8 CELLS × 768 CHANNELS</Tag>
      <Tag x={520} y={458}><tspan className="fig-dim">one cell ≈ 14 mm · every slice, ≤ 64 per series</tspan></Tag>
    </g>
  );
}

// ---------------------------------------------------------------- 05 box pooling (+ 07 explanation share the cells)
interface Cell {
  r: number;
  c: number;
  cx: number;
  cy: number;
  inBox: number; // share of the cell's 4 x 4 sub-points inside the box
  w: number; // normalized pooling weight
  contrib: number; // illustrative logit contribution
}

function useCells(finding: string): { cells: Cell[]; box: number[]; logit: number; bias: number } {
  return useMemo(() => {
    const [c0, r0, c1] = CROP;
    const cw = (c1 - c0) / 8;
    const b = SCHEMATIC_BOX[finding];
    const [hx, hy] = HOTSPOT[finding];
    const fi = FINDINGS.findIndex((f) => f.key === finding);
    const raw: Cell[] = [];
    for (let r = 0; r < 8; r++)
      for (let c = 0; c < 8; c++) {
        let inside = 0;
        for (let a = 0; a < 4; a++)
          for (let bb = 0; bb < 4; bb++) {
            const px = c0 + (c + (a + 0.5) / 4) * cw;
            const py = r0 + (r + (bb + 0.5) / 4) * cw;
            if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3]) inside++;
          }
        const cx = c0 + (c + 0.5) * cw;
        const cy = r0 + (r + 0.5) * cw;
        const d = Math.hypot(cx - hx, cy - hy);
        const att = Math.exp(-((d / 38) ** 2) + 0.35 * hash(r, c, fi));
        const cellLogit = 3.4 * Math.exp(-((d / 34) ** 2)) - 0.85 + 0.5 * (hash(c, r, fi + 3) - 0.5);
        raw.push({ r, c, cx, cy, inBox: inside / 16, w: (inside / 16) * att, contrib: cellLogit });
      }
    const total = raw.reduce((s, x) => s + x.w, 0) || 1;
    const cells = raw.map((x) => ({ ...x, w: x.w / total, contrib: (x.w / total) * x.contrib }));
    const bias = -1.6 - 0.2 * hash(fi, 9);
    const logit = cells.reduce((s, x) => s + x.contrib, 0) + bias;
    return { cells, box: b, logit, bias };
  }, [finding]);
}

function FigPool({ finding }: { finding: string }) {
  const { cells, box, logit } = useCells(finding);
  const X = 60, Y = 100, S = 360;
  const [c0, r0, c1] = CROP;
  const k = S / (c1 - c0);
  const maxW = Math.max(...cells.map((c) => c.w));
  const bx = [X + (box[0] - c0) * k, Y + (box[1] - r0) * k, X + (box[2] - c0) * k, Y + (box[3] - r0) * k];
  const p = 1 / (1 + Math.exp(-logit));
  const lit = cells.filter((c) => c.w > 0.002);
  const fx = 560, fy0 = 150, fy1 = 410;
  return (
    <g>
      <CropArt x={X} y={Y} size={S} id="crop-b" />
      {cells.map((c) => (
        <rect
          key={`${c.r}-${c.c}`}
          x={X + c.c * (S / 8) + 1}
          y={Y + c.r * (S / 8) + 1}
          width={S / 8 - 2}
          height={S / 8 - 2}
          className="cell"
          style={{ opacity: c.w > 0 ? 0.08 + 0.7 * (c.w / maxW) : 0 }}
        />
      ))}
      <Grid x={X} y={Y} size={S} cls="grid-line grid-line--static" />
      <rect x={Math.max(X, bx[0])} y={Math.max(Y, bx[1])} width={Math.min(X + S, bx[2]) - Math.max(X, bx[0])} height={Math.min(Y + S, bx[3]) - Math.max(Y, bx[1])} className="box-2d" />
      <Tag x={X} y={Y - 16}>{FINDINGS.find((f) => f.key === finding)?.name.toUpperCase()} · READS ONLY ITS BOX</Tag>
      {lit.map((c) => (
        <path
          key={`l${c.r}-${c.c}`}
          d={`M${X + (c.c + 0.5) * (S / 8)} ${Y + (c.r + 0.5) * (S / 8)} C ${fx - 90} ${Y + (c.r + 0.5) * (S / 8)}, ${fx - 60} ${(fy0 + fy1) / 2}, ${fx} ${(fy0 + fy1) / 2}`}
          className="flow"
          style={{ opacity: 0.08 + 0.6 * (c.w / maxW) }}
        />
      ))}
      <rect x={fx} y={fy0} width={14} height={fy1 - fy0} className="vec" />
      {Array.from({ length: 24 }, (_, i) => (
        <rect key={i} x={fx + 2} y={fy0 + 3 + i * ((fy1 - fy0 - 6) / 24)} width={10} height={(fy1 - fy0 - 6) / 24 - 2} className="vec-cell" style={{ opacity: 0.15 + 0.75 * hash(i, finding.length) }} />
      ))}
      <Tag x={fx - 4} y={fy0 - 14}>f̄ · 768</Tag>
      <line x1={fx + 26} x2={668} y1={(fy0 + fy1) / 2} y2={(fy0 + fy1) / 2} className="fig-arrow" markerEnd="url(#arrow)" />
      <g transform={`translate(680 ${(fy0 + fy1) / 2 - 44})`}>
        <Tag x={0} y={0}>h_k · f̄ + b_k</Tag>
        <Tag x={0} y={30}><tspan className="fig-dim">LOGIT</tspan></Tag>
        <text x={0} y={64} className="fig-num">{logit >= 0 ? "+" : "−"}{Math.abs(logit).toFixed(2)}</text>
        <Tag x={0} y={98}><tspan className="fig-dim">σ → PROBABILITY</tspan></Tag>
        <line x1={0} x2={200} y1={114} y2={114} className="fig-rule" />
        <line x1={0} x2={200 * p} y1={114} y2={114} className="fig-bar" />
        <Tag x={200} y={134} anchor="end" accent>{p.toFixed(2)}</Tag>
      </g>
      <Tag x={X} y={Y + S + 24}>
        <tspan className="fig-dim">weight = share of the cell inside the box × exp(attention) · normalized</tspan>
      </Tag>
      <Tag x={X} y={Y + S + 44}><tspan className="fig-dim">illustrative values</tspan></Tag>
    </g>
  );
}

// ---------------------------------------------------------------- 06 ensemble
function FigEnsemble({ finding, models }: { finding: string; models: number }) {
  const series = ["SAG PD FS", "SAG PD", "COR PD FS", "COR T1", "AX PD FS"];
  const fi = FINDINGS.findIndex((f) => f.key === finding);
  const M = Math.max(1, models);
  const vals = Array.from({ length: M }, (_, m) => series.map((_, s) => Math.min(0.97, Math.max(0.03, 0.62 + 0.28 * (hash(m, s, fi) - 0.5) + (s === 1 || s === 3 ? -0.14 : 0)))));
  const perModel = vals.map((r) => r.reduce((a, b) => a + b, 0) / r.length);
  const study = perModel.reduce((a, b) => a + b, 0) / M;
  const x0 = 170, dx = 82, y0 = 150, dy = Math.min(64, 300 / M);
  const mx = x0 + series.length * dx + 60;
  return (
    <g>
      {series.map((s, j) => (
        <text key={s} className="fig-tag fig-dim" transform={`translate(${x0 + j * dx + 4} ${y0 - 30}) rotate(-40)`}>{s}</text>
      ))}
      {vals.map((row, m) => (
        <g key={m} className="in" style={{ ["--i" as string]: m * 2 }}>
          <Tag x={40} y={y0 + m * dy + 4}>MODEL {m + 1}</Tag>
          {row.map((v, j) => (
            <g key={j}>
              <circle cx={x0 + j * dx} cy={y0 + m * dy} r={3 + 13 * v} className="ens-dot" />
              <text x={x0 + j * dx} y={y0 + m * dy + 26} className="fig-tag fig-dim" textAnchor="middle" style={{ fontSize: 9 }}>
                {v.toFixed(2)}
              </text>
            </g>
          ))}
          <line x1={x0 + (series.length - 1) * dx + 24} x2={mx - 24} y1={y0 + m * dy} y2={y0 + m * dy} className="fig-rule" />
          <circle cx={mx} cy={y0 + m * dy} r={3 + 13 * perModel[m]} className="ens-dot ens-dot--mean" />
          <text x={mx + 24} y={y0 + m * dy + 4} className="fig-tag">{perModel[m].toFixed(3)}</text>
        </g>
      ))}
      <Tag x={mx - 30} y={y0 - 40}>MEAN OVER SERIES</Tag>
      <line x1={mx + 80} x2={mx + 80} y1={y0 - 10} y2={y0 + (M - 1) * dy + 10} className="fig-rule" />
      <line x1={mx + 80} x2={mx + 110} y1={y0 + ((M - 1) * dy) / 2} y2={y0 + ((M - 1) * dy) / 2} className="fig-arrow" markerEnd="url(#arrow)" />
      <g transform={`translate(${mx + 122} ${y0 + ((M - 1) * dy) / 2 - 40})`}>
        <Tag x={0} y={0}>MEAN OVER MODELS</Tag>
        <text x={0} y={46} className="fig-num fig-num--accent">{study.toFixed(3)}</text>
        <Tag x={0} y={72}><tspan className="fig-dim">{FINDINGS[fi].name.toUpperCase()}</tspan></Tag>
      </g>
      <Tag x={40} y={y0 + M * dy + 40}>
        <tspan className="fig-dim">{M} fold checkpoints loaded on this machine · equal weights, fixed before any test score · illustrative values</tspan>
      </Tag>
    </g>
  );
}

// ---------------------------------------------------------------- 07 explanation
function FigExplain({ finding }: { finding: string }) {
  const { cells, logit, bias } = useCells(finding);
  const X = 60, Y = 100, S = 360, cs = S / 8;
  const max = Math.max(...cells.map((c) => Math.abs(c.contrib)), 1e-6);
  const ranked = [...cells].filter((c) => c.w > 0).sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
  const top = ranked.slice(0, 5);
  const rest = ranked.slice(5).reduce((s, c) => s + c.contrib, 0);
  const sign = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
  return (
    <g>
      <CropArt x={X} y={Y} size={S} id="crop-c" />
      <Grid x={X} y={Y} size={S} cls="grid-line grid-line--static" />
      {cells.map((c) => {
        if (c.w <= 0) return null;
        const f = Math.sqrt(Math.abs(c.contrib) / max);
        const s = Math.max(3, (cs - 8) * f);
        return (
          <rect
            key={`${c.r}-${c.c}`}
            x={X + (c.c + 0.5) * cs - s / 2}
            y={Y + (c.r + 0.5) * cs - s / 2}
            width={s}
            height={s}
            className={c.contrib >= 0 ? "contrib contrib--pos" : "contrib contrib--neg"}
          />
        );
      })}
      <Tag x={X} y={Y - 16}>CONTRIBUTION OF EACH CELL TO THE LOGIT</Tag>
      <g transform="translate(500 120)">
        <text x={0} y={0} className="fig-formula">logit_k = Σ_cells share · (h_k · f) + b_k</text>
        <line x1={0} x2={380} y1={22} y2={22} className="fig-rule" />
        {top.map((c, i) => (
          <g key={i} transform={`translate(0 ${50 + i * 26})`}>
            <Tag x={0} y={0}><tspan className="fig-dim">CELL r{c.r} c{c.c}</tspan></Tag>
            <Tag x={170} y={0}><tspan className="fig-dim">share {c.w.toFixed(3)}</tspan></Tag>
            <Tag x={380} y={0} anchor="end" accent={c.contrib >= 0}>{sign(c.contrib)}</Tag>
          </g>
        ))}
        <g transform={`translate(0 ${50 + top.length * 26})`}>
          <Tag x={0} y={0}><tspan className="fig-dim">{ranked.length - top.length} OTHER CELLS IN THE BOX</tspan></Tag>
          <Tag x={380} y={0} anchor="end">{sign(rest)}</Tag>
          <Tag x={0} y={26}><tspan className="fig-dim">BIAS b_k</tspan></Tag>
          <Tag x={380} y={26} anchor="end">{sign(bias)}</Tag>
          <line x1={0} x2={380} y1={40} y2={40} className="fig-rule" />
          <Tag x={0} y={64}>= LOGIT</Tag>
          <Tag x={380} y={64} anchor="end" accent>{sign(logit)}</Tag>
          <Tag x={0} y={104}><tspan className="fig-dim">exact: the head is linear and the pooling is a weighted mean,</tspan></Tag>
          <Tag x={0} y={122}><tspan className="fig-dim">so the terms add up to the logit. The workstation checks it</tspan></Tag>
          <Tag x={0} y={140}><tspan className="fig-dim">on every run. Illustrative values here.</tspan></Tag>
        </g>
      </g>
      <g transform={`translate(${X} ${Y + S + 26})`}>
        <rect x={0} y={-9} width={10} height={10} className="contrib contrib--pos" />
        <Tag x={18} y={0}><tspan className="fig-dim">for the finding</tspan></Tag>
        <rect x={140} y={-9} width={10} height={10} className="contrib contrib--neg" />
        <Tag x={158} y={0}><tspan className="fig-dim">against</tspan></Tag>
        <Tag x={240} y={0}><tspan className="fig-dim">area ∝ |contribution|</tspan></Tag>
      </g>
    </g>
  );
}

export default function MethodFigure({ step, progress, finding, models }: { step: number; progress: number; finding: string; models: number }) {
  const layers = [
    <FigStudy progress={step === 0 ? progress : 0} />,
    <FigSide on={step === 1} />,
    <FigBoxes finding={finding} />,
    <FigCrop progress={step === 3 ? progress : 0} />,
    <FigPool finding={finding} />,
    <FigEnsemble finding={finding} models={models} />,
    <FigExplain finding={finding} />,
  ];
  return (
    <svg className="method-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pipeline figure">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10" fill="none" stroke="currentColor" strokeWidth="1.2" />
        </marker>
      </defs>
      {layers.map((l, i) => (
        <g key={i} className={i === step ? "layer on" : "layer"} aria-hidden={i !== step}>
          {l}
        </g>
      ))}
    </svg>
  );
}
