// The 3D viewport. Every series sits in one scene at its true patient position (from ImagePositionPatient /
// ImageOrientationPatient), so the sagittal, coronal and axial stacks intersect where they really do. The active series
// is ray-marched as a volume; the others show their current slice. Model outputs (ROI boxes, the classifier's crop,
// its feature cells and their attribution) are drawn in each series' own normalized coordinates, i.e. where the
// model computed them.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SeriesMeta } from "../lib/api";
import { lpsToScene, studyCentre, toPatient, unitToScene, type Geo, type V3 } from "./geometry";

const YELLOW = new THREE.Color("#f5b700");
const WHITE = new THREE.Color("#f2f1ec");

const volumeVert = /* glsl */ `
varying vec3 vPos;
void main() {
  vPos = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const volumeFrag = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uVol;
uniform vec3 uCam;          // camera position in the unit cube's coordinates
uniform float uLo, uHi;     // window on the display values (0..1)
uniform float uDensity;
uniform int uMode;          // 0 DVR, 1 MIP
uniform float uCut;         // w above which the volume is cut away (> 1: no cut)
uniform vec3 uTint;
uniform vec3 uTexel;        // one voxel in unit-cube coordinates
uniform vec3 uExtent;       // the volume's size in mm (anisotropic voxels)
varying vec3 vPos;
out highp vec4 outColor;

vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(0.0) - o) * inv, t1 = (vec3(1.0) - o) * inv;
  vec3 tn = min(t0, t1), tf = max(t0, t1);
  return vec2(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

void main() {
  vec3 d = normalize(vPos - uCam);
  vec2 t = hitBox(uCam, d);
  t.x = max(t.x, 0.0);
  if (t.x >= t.y) discard;
  const int STEPS = 320;
  float dt = 1.7320508 / float(STEPS);
  float s = t.x + dt * hash(gl_FragCoord.xy);
  vec4 acc = vec4(0.0);
  float mip = 0.0;
  for (int i = 0; i < STEPS; i++) {
    if (s > t.y) break;
    vec3 p = uCam + d * s;
    if (p.z <= uCut) {
      float v = texture(uVol, p).r;
      float x = clamp((v - uLo) / max(uHi - uLo, 1e-3), 0.0, 1.0);
      if (uMode == 1) {
        mip = max(mip, x);
      } else {
        float a = pow(x, 2.6) * uDensity;
        a = 1.0 - pow(1.0 - clamp(a, 0.0, 1.0), dt * 220.0);
        vec3 c = mix(uTint * 0.55, uTint, x);
        if (a > 0.004) {
          // central-difference gradient, in mm, for headlight shading (depth cue without colour)
          vec3 g = vec3(
            texture(uVol, p + vec3(uTexel.x, 0.0, 0.0)).r - texture(uVol, p - vec3(uTexel.x, 0.0, 0.0)).r,
            texture(uVol, p + vec3(0.0, uTexel.y, 0.0)).r - texture(uVol, p - vec3(0.0, uTexel.y, 0.0)).r,
            texture(uVol, p + vec3(0.0, 0.0, uTexel.z)).r - texture(uVol, p - vec3(0.0, 0.0, uTexel.z)).r
          ) / (2.0 * uTexel * uExtent);
          float gl = length(g);
          if (gl > 1e-4) {
            vec3 n = g / gl;
            vec3 vd = normalize(d * uExtent);
            float diff = abs(dot(n, vd));
            float edge = clamp(gl * 18.0, 0.0, 1.0);
            c *= mix(1.0, 0.32 + 0.78 * diff, edge);
          }
        }
        acc.rgb += (1.0 - acc.a) * a * c;
        acc.a += (1.0 - acc.a) * a;
        if (acc.a > 0.97) break;
      }
    }
    s += dt;
  }
  if (uMode == 1) {
    float m = pow(mip, 1.25);
    outColor = vec4(uTint * m, m);
  } else {
    outColor = acc;
  }
}`;

const sliceFrag = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uVol;
uniform float uW, uLo, uHi, uOpacity;
uniform vec3 uTint;
varying vec2 vUv;
out highp vec4 outColor;
void main() {
  float v = texture(uVol, vec3(vUv, uW)).r;
  float x = clamp((v - uLo) / max(uHi - uLo, 1e-3), 0.0, 1.0);
  outColor = vec4(uTint * x, uOpacity * (0.25 + 0.75 * x));
}`;

const sliceVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const cellVert = /* glsl */ `
attribute float aAlpha;
varying float vA;
void main() {
  vA = aAlpha;
  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
}`;
const cellFrag = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vA;
void main() {
  if (vA < 0.004) discard;
  gl_FragColor = vec4(uColor, vA * uOpacity);
}`;

function unitBoxEdges(): THREE.BufferGeometry {
  const g = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
  g.translate(0.5, 0.5, 0.5);
  return g;
}
const UNIT_EDGES = unitBoxEdges();

interface SeriesObj {
  meta: SeriesMeta;
  geo: Geo;
  group: THREE.Group;
  outline: THREE.LineSegments;
  outlineMat: THREE.LineBasicMaterial;
  tex?: THREE.Data3DTexture;
  volume?: THREE.Mesh;
  volumeMat?: THREE.ShaderMaterial;
  slice?: THREE.Mesh;
  sliceMat?: THREE.ShaderMaterial;
  boxes: THREE.Group; // 12 boxes (mean), plus TTA passes
  crop: THREE.LineSegments;
  cells: THREE.InstancedMesh;
  cellAlpha: THREE.InstancedBufferAttribute;
  cellMat: THREE.ShaderMaterial;
  lattice: THREE.InstancedMesh;
  latticeAlpha: THREE.InstancedBufferAttribute;
  latticeMat: THREE.ShaderMaterial;
}

export interface SceneBox {
  box: number[]; // normalized x0 y0 z0 x1 y1 z1 (z already mapped to w)
  color: "accent" | "white";
  opacity: number;
  dashed?: boolean;
}

export interface SeriesView {
  visible: boolean;
  emphasis: number; // 0 dim .. 1 focus
  slice: number;
  showSlice: boolean;
  boxes: SceneBox[];
  crop: number[] | null; // x0 y0 x1 y1
  cropOpacity: number;
  cells: { u0: number; v0: number; u1: number; v1: number; ws: number[]; alpha: Float32Array; h: number; w: number } | null;
  lattice: { u0: number; v0: number; u1: number; v1: number; ws: number[]; h: number; w: number; opacity: number } | null;
}

export interface SceneState {
  active: string | null;
  mode: "dvr" | "mip" | "slices";
  cut: boolean;
  window: [number, number];
  series: Record<string, SeriesView>;
  point: V3 | null;
  cellColor: "accent" | "white"; // evidence (yellow) or pooling weight (white)
}

const MAX_CELLS = 64 * 64;

export class Scene3D {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  root = new THREE.Group();
  objs = new Map<string, SeriesObj>();
  centre: V3 = [0, 0, 0];
  radius = 120;
  private raf = 0;
  private dirty = true;
  private tween: { from: THREE.Vector3; to: THREE.Vector3; tFrom: THREE.Vector3; tTo: THREE.Vector3; t0: number; dur: number } | null = null;
  private pointMesh: THREE.Group;
  private ro: ResizeObserver;
  onFrame?: (cam: THREE.PerspectiveCamera) => void;
  state: SceneState | null = null;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.renderer.setClearColor(0x000000, 1);
    host.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(28, 1, 1, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.9;
    this.controls.addEventListener("change", () => (this.dirty = true));
    this.scene.add(this.root);
    // crosshair: three short axis lines
    this.pointMesh = new THREE.Group();
    const m = new THREE.LineBasicMaterial({ color: YELLOW, transparent: true, opacity: 0.9, depthTest: false });
    for (const a of [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]) {
      const g = new THREE.BufferGeometry().setFromPoints([a.clone().multiplyScalar(-7), a.clone().multiplyScalar(7)]);
      const l = new THREE.Line(g, m);
      l.renderOrder = 20;
      this.pointMesh.add(l);
    }
    this.pointMesh.visible = false;
    this.scene.add(this.pointMesh);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      this.frame();
    };
    loop();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls.dispose();
    for (const o of this.objs.values()) this.disposeObj(o);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private disposeObj(o: SeriesObj) {
    o.tex?.dispose();
    o.volumeMat?.dispose();
    o.sliceMat?.dispose();
    o.cellMat.dispose();
    o.latticeMat.dispose();
    o.cells.geometry.dispose();
    o.lattice.geometry.dispose();
    this.root.remove(o.group);
  }

  invalidate() {
    this.dirty = true;
  }

  resize() {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /** Builds one object per series; keeps the camera if the study is the same. */
  setStudy(series: SeriesMeta[], geos: Record<string, Geo>) {
    for (const o of this.objs.values()) this.disposeObj(o);
    this.objs.clear();
    const gs = series.map((s) => geos[s.uid]).filter(Boolean);
    this.centre = studyCentre(gs);
    let r = 60;
    for (const s of series) {
      const geo = geos[s.uid];
      if (!geo) continue;
      for (const c of [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]] as V3[]) {
        r = Math.max(r, lpsToScene(toPatient(geo, c[0], c[1], c[2]), this.centre).length());
      }
      const group = new THREE.Group();
      group.matrixAutoUpdate = false;
      group.matrix.copy(unitToScene(geo, this.centre));
      const outlineMat = new THREE.LineBasicMaterial({ color: WHITE, transparent: true, opacity: 0.16, depthWrite: false });
      const outline = new THREE.LineSegments(UNIT_EDGES, outlineMat);
      group.add(outline);
      const boxes = new THREE.Group();
      group.add(boxes);
      const crop = new THREE.LineSegments(UNIT_EDGES, new THREE.LineDashedMaterial({ color: YELLOW, transparent: true, opacity: 0, dashSize: 0.02, gapSize: 0.015, depthTest: false }));
      crop.computeLineDistances();
      crop.renderOrder = 12;
      group.add(crop);
      const mkCells = (color: THREE.Color, inset: number) => {
        const geom = new THREE.PlaneGeometry(inset, inset);
        const alpha = new THREE.InstancedBufferAttribute(new Float32Array(MAX_CELLS), 1);
        alpha.setUsage(THREE.DynamicDrawUsage);
        geom.setAttribute("aAlpha", alpha);
        const mat = new THREE.ShaderMaterial({
          vertexShader: cellVert,
          fragmentShader: cellFrag,
          uniforms: { uColor: { value: color.clone() }, uOpacity: { value: 1 } },
          transparent: true,
          depthWrite: false,
          depthTest: false,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.InstancedMesh(geom, mat, MAX_CELLS);
        mesh.count = 0;
        mesh.frustumCulled = false;
        return { mesh, alpha, mat };
      };
      const cells = mkCells(YELLOW, 0.86);
      cells.mesh.renderOrder = 14;
      const lattice = mkCells(WHITE, 0.8);
      lattice.mesh.renderOrder = 13;
      group.add(lattice.mesh, cells.mesh);
      this.root.add(group);
      this.objs.set(s.uid, {
        meta: s, geo, group, outline, outlineMat, boxes, crop,
        cells: cells.mesh, cellAlpha: cells.alpha, cellMat: cells.mat,
        lattice: lattice.mesh, latticeAlpha: lattice.alpha, latticeMat: lattice.mat,
      });
    }
    this.radius = r;
    this.home(false);
    this.dirty = true;
  }

  setVolume(uid: string, vol: { data: Uint8Array; d: number; h: number; w: number }) {
    const o = this.objs.get(uid);
    if (!o || o.tex) return;
    const tex = new THREE.Data3DTexture(vol.data, vol.w, vol.h, vol.d);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    o.tex = tex;
    const box = new THREE.BoxGeometry(1, 1, 1);
    box.translate(0.5, 0.5, 0.5);
    o.volumeMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: volumeVert,
      fragmentShader: volumeFrag,
      uniforms: {
        uVol: { value: tex }, uCam: { value: new THREE.Vector3() }, uLo: { value: 0.08 }, uHi: { value: 0.9 },
        uDensity: { value: 0.11 }, uMode: { value: 0 }, uCut: { value: 2 }, uTint: { value: new THREE.Color("#f4efe3") },
        uTexel: { value: new THREE.Vector3(1 / vol.w, 1 / vol.h, 1 / vol.d) }, uExtent: { value: new THREE.Vector3(...o.geo.size) },
      },
      transparent: true,
      side: THREE.BackSide,
      depthWrite: false,
    });
    o.volume = new THREE.Mesh(box, o.volumeMat);
    o.volume.renderOrder = 2;
    o.volume.visible = false;
    o.group.add(o.volume);
    const plane = new THREE.PlaneGeometry(1, 1);
    plane.translate(0.5, 0.5, 0);
    o.sliceMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: sliceVert,
      fragmentShader: sliceFrag,
      uniforms: { uVol: { value: tex }, uW: { value: 0.5 }, uLo: { value: 0 }, uHi: { value: 1 }, uOpacity: { value: 0.9 }, uTint: { value: new THREE.Color("#f4efe3") } },
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    o.slice = new THREE.Mesh(plane, o.sliceMat);
    o.slice.renderOrder = 3;
    o.group.add(o.slice);
    this.dirty = true;
    if (this.state) this.apply(this.state);
  }

  /** Applies the full view state (cheap: uniforms, instance attributes, visibility). */
  apply(st: SceneState) {
    this.state = st;
    for (const [uid, o] of this.objs) {
      const v = st.series[uid];
      const vis = !!v?.visible;
      o.group.visible = vis;
      if (!v || !vis) continue;
      const isActive = uid === st.active;
      o.outlineMat.opacity = isActive ? 0.42 : 0.03 + 0.45 * v.emphasis * v.emphasis;
      o.outlineMat.color.copy(v.emphasis > 0.75 ? YELLOW : WHITE);
      const w = (v.slice + 0.5) / o.geo.depth;
      if (o.volume && o.volumeMat) {
        o.volume.visible = isActive && st.mode !== "slices";
        o.volumeMat.uniforms.uMode.value = st.mode === "mip" ? 1 : 0;
        o.volumeMat.uniforms.uLo.value = st.window[0];
        o.volumeMat.uniforms.uHi.value = st.window[1];
        o.volumeMat.uniforms.uCut.value = st.cut ? w : 2;
      }
      if (o.slice && o.sliceMat) {
        o.slice.visible = v.showSlice && (st.mode === "slices" || !isActive || st.cut);
        o.slice.position.z = w;
        o.sliceMat.uniforms.uW.value = w;
        o.sliceMat.uniforms.uOpacity.value = isActive ? 0.95 : 0.25 + 0.55 * v.emphasis;
      }
      this.setBoxes(o, v.boxes);
      // crop prism
      const cm = o.crop.material as THREE.LineDashedMaterial;
      if (v.crop) {
        const [x0, y0, x1, y1] = v.crop;
        o.crop.position.set(x0, y0, 0);
        o.crop.scale.set(x1 - x0, y1 - y0, 1);
        cm.opacity = v.cropOpacity;
        o.crop.visible = v.cropOpacity > 0.01;
      } else o.crop.visible = false;
      this.setCells(o.cells, o.cellAlpha, v.cells ? { ...v.cells, alpha: v.cells.alpha } : null);
      (o.cellMat.uniforms.uColor.value as THREE.Color).copy(st.cellColor === "white" ? WHITE : YELLOW);
      if (v.lattice) {
        const n = v.lattice.ws.length * v.lattice.h * v.lattice.w;
        const a = new Float32Array(n).fill(1);
        this.setCells(o.lattice, o.latticeAlpha, { ...v.lattice, alpha: a });
        o.latticeMat.uniforms.uOpacity.value = v.lattice.opacity;
      } else o.lattice.count = 0;
    }
    if (st.point) {
      this.pointMesh.visible = true;
      this.pointMesh.position.copy(lpsToScene(st.point, this.centre));
    } else this.pointMesh.visible = false;
    this.dirty = true;
  }

  private setCells(mesh: THREE.InstancedMesh, attr: THREE.InstancedBufferAttribute, c: SeriesView["cells"]) {
    if (!c) {
      mesh.count = 0;
      return;
    }
    const m = new THREE.Matrix4();
    const cw = (c.u1 - c.u0) / c.w;
    const ch = (c.v1 - c.v0) / c.h;
    let n = 0;
    const arr = attr.array as Float32Array;
    for (let k = 0; k < c.ws.length; k++) {
      for (let r = 0; r < c.h; r++) {
        for (let q = 0; q < c.w; q++) {
          const a = c.alpha[(k * c.h + r) * c.w + q];
          if (a < 0.004 || n >= MAX_CELLS) continue;
          m.makeScale(cw, ch, 1).setPosition(c.u0 + (q + 0.5) * cw, c.v0 + (r + 0.5) * ch, c.ws[k]);
          mesh.setMatrixAt(n, m);
          arr[n] = a;
          n++;
        }
      }
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    attr.needsUpdate = true;
  }

  private setBoxes(o: SeriesObj, boxes: SceneBox[]) {
    const kids = o.boxes.children as THREE.LineSegments[];
    while (kids.length < boxes.length) {
      const l = new THREE.LineSegments(UNIT_EDGES, new THREE.LineDashedMaterial({ transparent: true, depthTest: false, dashSize: 0.012, gapSize: 0.012 }));
      l.computeLineDistances();
      l.renderOrder = 10;
      o.boxes.add(l);
    }
    kids.forEach((l, i) => {
      const b = boxes[i];
      l.visible = !!b;
      if (!b) return;
      const mat = l.material as THREE.LineDashedMaterial;
      mat.color.copy(b.color === "accent" ? YELLOW : WHITE);
      mat.opacity = b.opacity;
      // LineDashedMaterial draws solid when the gap is zero
      mat.gapSize = b.dashed ? 0.012 : 0;
      mat.dashSize = b.dashed ? 0.012 : 1;
      l.position.set(b.box[0], b.box[1], b.box[2]);
      l.scale.set(Math.max(1e-3, b.box[3] - b.box[0]), Math.max(1e-3, b.box[4] - b.box[1]), Math.max(1e-3, b.box[5] - b.box[2]));
    });
  }

  /** Default camera: in front of the knee, a little lateral and above. */
  home(animate = true) {
    const d = this.radius * 3.0;
    const pos = new THREE.Vector3(0.55, 0.38, 1).normalize().multiplyScalar(d);
    this.moveTo(pos, new THREE.Vector3(0, 0, 0), animate);
  }

  /** Flies to a box of one series (normalized coordinates). */
  focusBox(uid: string, box: number[]) {
    const o = this.objs.get(uid);
    if (!o) return;
    const corners = [] as THREE.Vector3[];
    for (const c of [[box[0], box[1], box[2]], [box[3], box[4], box[5]]]) {
      corners.push(lpsToScene(toPatient(o.geo, c[0], c[1], c[2]), this.centre));
    }
    const centre = corners[0].clone().add(corners[1]).multiplyScalar(0.5);
    const size = Math.max(30, corners[0].distanceTo(corners[1]));
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.moveTo(centre.clone().add(dir.multiplyScalar(size * 3.6)), centre, true);
  }

  private moveTo(pos: THREE.Vector3, target: THREE.Vector3, animate: boolean) {
    if (!animate) {
      this.camera.position.copy(pos);
      this.controls.target.copy(target);
      this.controls.update();
      this.dirty = true;
      return;
    }
    this.tween = { from: this.camera.position.clone(), to: pos, tFrom: this.controls.target.clone(), tTo: target, t0: performance.now(), dur: 1100 };
  }

  /** Scene direction of a patient axis (for the orientation gizmo). */
  projectAxis(v: V3): [number, number] {
    const a = new THREE.Vector3(v[0], v[2], -v[1]).transformDirection(this.camera.matrixWorldInverse);
    return [a.x, -a.y];
  }

  private frame() {
    if (this.tween) {
      const k = Math.min(1, (performance.now() - this.tween.t0) / this.tween.dur);
      const e = 1 - Math.pow(1 - k, 4);
      this.camera.position.lerpVectors(this.tween.from, this.tween.to, e);
      this.controls.target.lerpVectors(this.tween.tFrom, this.tween.tTo, e);
      if (k >= 1) this.tween = null;
      this.dirty = true;
    }
    const moved = this.controls.update();
    if (!this.dirty && !moved) return;
    this.dirty = false;
    // camera position in each active volume's unit cube
    for (const o of this.objs.values()) {
      if (o.volume?.visible && o.volumeMat) {
        o.group.updateMatrixWorld(true);
        const inv = new THREE.Matrix4().copy(o.group.matrixWorld).invert();
        (o.volumeMat.uniforms.uCam.value as THREE.Vector3).copy(this.camera.position).applyMatrix4(inv);
      }
    }
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.(this.camera);
  }
}
