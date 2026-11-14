// Series geometry in patient coordinates (DICOM LPS, millimetres) and the mapping to the 3D scene.
//
// A series' "unit cube" (u, v, w) in [0, 1]^3 spans its whole volume: u over columns, v over rows, w over slices
// (slice k's centre at w = (k + 0.5) / D). The ROI detector and the classifier both work in these normalized native
// coordinates, so boxes and feature cells are placed exactly where the model put them.
import * as THREE from "three";
import type { SeriesMeta } from "../lib/api";

export type V3 = [number, number, number];

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);

export interface Geo {
  uid: string;
  X: V3; // direction of increasing column (DICOM row direction)
  Y: V3; // direction of increasing row
  N: V3; // slice normal (X x Y)
  dc: number; // column spacing mm
  dr: number; // row spacing mm
  step: V3; // vector between consecutive slice centres
  first: V3; // ImagePositionPatient of slice 0 (centre of its first pixel)
  rows: number;
  cols: number;
  depth: number;
  origin: V3; // corner of the unit cube (u = v = w = 0)
  size: V3; // edge vectors' lengths (mm)
}

export function geometry(s: SeriesMeta): Geo {
  const X = s.iop.slice(0, 3) as V3;
  const Y = s.iop.slice(3, 6) as V3;
  const N = cross(X, Y);
  const [dr, dc] = s.pixelSpacing;
  const first = s.positions[0] as V3;
  const last = s.positions[s.positions.length - 1] as V3;
  const D = s.positions.length;
  const step: V3 = D > 1 ? mul(sub(last, first), 1 / (D - 1)) : mul(N, s.sliceSpacing || 3);
  const origin = sub(sub(sub(first, mul(X, 0.5 * dc)), mul(Y, 0.5 * dr)), mul(step, 0.5));
  return { uid: s.uid, X, Y, N, dc, dr, step, first, rows: s.rows, cols: s.cols, depth: D, origin, size: [s.cols * dc, s.rows * dr, norm(step) * D] };
}

/** (u, v, w) in the unit cube -> patient LPS mm. */
export function toPatient(g: Geo, u: number, v: number, w: number): V3 {
  return add(add(add(g.origin, mul(g.X, u * g.cols * g.dc)), mul(g.Y, v * g.rows * g.dr)), mul(g.step, w * g.depth));
}

/** Patient point -> (u, v, w); w maps slice k's centre to (k + 0.5) / D. */
export function toUnit(g: Geo, p: V3): V3 {
  const d = sub(p, g.origin);
  const sn = dot(g.step, g.N);
  return [dot(d, g.X) / (g.cols * g.dc), dot(d, g.Y) / (g.rows * g.dr), dot(d, g.N) / (sn * g.depth)];
}

export const sliceOfW = (g: Geo, w: number) => Math.min(g.depth - 1, Math.max(0, Math.round(w * g.depth - 0.5)));
export const wOfSlice = (g: Geo, k: number) => (k + 0.5) / g.depth;

/** Patient LPS -> scene: x = patient left, y = superior, z = anterior (camera in front of the knee). */
export function lpsToScene(p: V3, centre: V3): THREE.Vector3 {
  return new THREE.Vector3(p[0] - centre[0], p[2] - centre[2], -(p[1] - centre[1]));
}

/** Matrix taking the unit cube to scene coordinates. */
export function unitToScene(g: Geo, centre: V3): THREE.Matrix4 {
  const o = lpsToScene(g.origin, centre);
  const ex = lpsToScene(add(centre, mul(g.X, g.cols * g.dc)), centre);
  const ey = lpsToScene(add(centre, mul(g.Y, g.rows * g.dr)), centre);
  const ez = lpsToScene(add(centre, mul(g.step, g.depth)), centre);
  return new THREE.Matrix4().makeBasis(ex, ey, ez).setPosition(o);
}

export function studyCentre(geos: Geo[]): V3 {
  let c: V3 = [0, 0, 0];
  for (const g of geos) c = add(c, toPatient(g, 0.5, 0.5, 0.5));
  return mul(c, 1 / Math.max(1, geos.length));
}

/** Anatomical letter for a direction in LPS (dominant axis). */
export function letter(v: V3): string {
  const a = [Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2])];
  const i = a.indexOf(Math.max(...a));
  return i === 0 ? (v[0] > 0 ? "L" : "R") : i === 1 ? (v[1] > 0 ? "P" : "A") : v[2] > 0 ? "S" : "I";
}

/** Edge letters of a displayed slice: [top, right, bottom, left]. */
export function edgeLetters(g: Geo): [string, string, string, string] {
  return [letter(mul(g.Y, -1)), letter(g.X), letter(g.Y), letter(mul(g.X, -1))];
}

export function sliceLocation(g: Geo, k: number): number {
  return dot(add(g.first, mul(g.step, k)), g.N);
}

export const vec = { sub, add, mul, dot, cross, norm };
