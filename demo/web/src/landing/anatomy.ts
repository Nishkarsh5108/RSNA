// Schematic knee anatomy for the method figures (line art in a 256 x 256 image frame) and an isometric helper.
// These drawings explain the pipeline; they are not data. The workstation shows the model's real boxes and maps.

export const SAG = {
  // anterior on the left, as a sagittal knee MRI is displayed
  skinA: "M38 0 C46 60 42 112 50 150 C56 190 70 222 76 256",
  skinP: "M226 0 C232 70 236 130 232 170 C228 208 216 238 208 256",
  femur: "M112 0 L113 62 C113 86 96 96 90 116 C84 140 100 158 126 160 C150 162 182 158 192 136 C200 116 186 96 168 84 L165 0 Z",
  patella: "M64 86 C74 82 84 92 84 110 C84 128 76 138 66 136 C58 134 56 120 57 106 C58 94 58 88 64 86 Z",
  tibia: "M86 170 C96 166 150 165 194 168 C198 176 192 188 178 196 C172 210 168 230 166 256 L108 256 C108 232 104 212 98 200 C88 192 82 180 86 170 Z",
  menA: "M88 168 L112 166 L95 158 Z",
  menP: "M166 166 L195 168 L189 155 Z",
  acl: "M118 166 C128 150 142 136 156 124",
  ptendon: "M68 136 C76 160 88 182 100 200",
  qtendon: "M66 86 C72 58 88 36 100 0",
};

export const COR = {
  // a right knee, radiological convention: the patient's right (lateral side, fibula) on the viewer's left
  skin: "M30 0 C26 80 30 150 40 200 C46 230 50 256 50 256 M226 0 C230 80 226 150 216 200 C210 230 206 256 206 256",
  femur: "M110 0 L108 70 C90 82 70 96 66 120 C64 140 76 150 94 150 C108 150 114 140 120 132 C124 128 132 128 136 132 C142 140 148 150 162 150 C182 150 194 140 192 120 C188 96 166 82 148 70 L146 0 Z",
  tibia: "M64 164 C80 158 108 160 128 162 C148 160 176 158 192 164 C194 180 180 194 166 200 C160 220 158 240 158 256 L100 256 C100 240 98 220 92 200 C78 194 64 180 64 164 Z",
  fibula: "M50 188 C58 182 68 186 68 196 L66 256 L56 256 L54 200 C48 196 46 192 50 188 Z",
  menL: "M64 160 L100 158 L70 150 Z",
  menM: "M156 158 L192 160 L186 150 Z",
  mcl: "M198 98 C202 130 202 162 196 204",
  lcl: "M60 112 C52 140 52 168 56 186",
};

export const AX = {
  skin: "M128 14 C190 14 236 62 236 128 C236 196 190 244 128 244 C66 244 20 196 20 128 C20 62 66 14 128 14 Z",
  femur: "M70 124 C70 84 100 68 128 76 C156 68 186 84 186 124 C186 164 160 194 140 188 C136 176 120 176 116 188 C96 194 70 164 70 124 Z",
  patella: "M98 52 C112 36 144 36 158 52 C150 64 106 64 98 52 Z",
};

/** Per finding: a schematic box in the sagittal image (x0, y0, x1, y1 in px) and its slice range (0 lateral .. 1 medial). */
export const SCHEMATIC_BOX: Record<string, [number, number, number, number, number, number]> = {
  ACL: [114, 118, 166, 170, 0.38, 0.62],
  MCL: [92, 104, 196, 206, 0.82, 1.0],
  "Medial Meniscus": [86, 150, 198, 178, 0.56, 0.92],
  "Lateral Meniscus": [86, 150, 198, 178, 0.08, 0.44],
  "Medial OA": [82, 112, 200, 196, 0.52, 0.96],
  "Lateral OA": [82, 112, 200, 196, 0.04, 0.48],
  "PF OA": [50, 76, 124, 146, 0.25, 0.75],
  Effusion: [48, 30, 206, 200, 0.1, 0.9],
  Synovitis: [56, 96, 132, 196, 0.15, 0.85],
  "Baker's": [182, 126, 236, 206, 0.55, 0.95],
  Contusion: [78, 96, 204, 222, 0.05, 0.95],
  Fracture: [56, 60, 208, 250, 0.0, 1.0],
};

/** Where in the schematic each finding's evidence would sit (for the illustrative attention / contribution maps). */
export const HOTSPOT: Record<string, [number, number]> = {
  ACL: [138, 144],
  MCL: [150, 170],
  "Medial Meniscus": [182, 162],
  "Lateral Meniscus": [100, 164],
  "Medial OA": [150, 164],
  "Lateral OA": [120, 164],
  "PF OA": [84, 110],
  Effusion: [90, 64],
  Synovitis: [92, 160],
  "Baker's": [212, 168],
  Contusion: [120, 140],
  Fracture: [140, 196],
};

/** The 112 mm crop of the 140 mm frame about the union of the ACL, meniscus and tibiofemoral-OA boxes (as crop_theta). */
export function cropOf(): [number, number, number, number] {
  const keys = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"];
  const x0 = Math.min(...keys.map((k) => SCHEMATIC_BOX[k][0]));
  const y0 = Math.min(...keys.map((k) => SCHEMATIC_BOX[k][1]));
  const x1 = Math.max(...keys.map((k) => SCHEMATIC_BOX[k][2]));
  const y1 = Math.max(...keys.map((k) => SCHEMATIC_BOX[k][3]));
  const half = (256 * (112 / 140)) / 2;
  const cx = Math.min(Math.max((x0 + x1) / 2, half), 256 - half);
  const cy = Math.min(Math.max((y0 + y1) / 2, half), 256 - half);
  return [cx - half, cy - half, cx + half, cy + half];
}

export type V3 = [number, number, number];

export function makeIso(cx: number, cy: number, s: number) {
  return (p: V3): [number, number] => [cx + (p[0] - p[1]) * s * 0.866, cy + (p[0] + p[1]) * s * 0.5 - p[2] * s];
}

export function pts(list: [number, number][]) {
  return list.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

/** SVG matrix that maps image pixels (0..256) onto a plane: origin = image (0,0), u / v = 3D directions of +x / +y. */
export function planeMatrix(iso: (p: V3) => [number, number], origin: V3, u: V3, v: V3) {
  const o = iso(origin);
  const a = iso([origin[0] + u[0], origin[1] + u[1], origin[2] + u[2]]);
  const b = iso([origin[0] + v[0], origin[1] + v[1], origin[2] + v[2]]);
  return `matrix(${(a[0] - o[0]) / 256} ${(a[1] - o[1]) / 256} ${(b[0] - o[0]) / 256} ${(b[1] - o[1]) / 256} ${o[0]} ${o[1]})`;
}

/** Deterministic pseudo-random in [0, 1) for illustrative values. */
export function hash(i: number, j: number, k = 0) {
  const s = Math.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453;
  return s - Math.floor(s);
}
