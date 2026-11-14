// A field of white dots that behaves like ferrofluid under a magnet: where the field is strong the surface breaks into
// a hexagonal lattice of sharp peaks (the Rosensweig instability). The cursor is the magnet. Everything is computed in
// the vertex shader from a few uniforms; the CPU only integrates the magnet (position, strength, a short decaying trail).
import * as THREE from "three";

const TRAIL = 7;

const NOISE = /* glsl */ `
// Simplex 3D noise, Ashima Arts / Stefan Gustavson (MIT).
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;

const COMMON = /* glsl */ `
uniform float uTime;
uniform vec3 uMag[${TRAIL}];   // x, z, strength (the magnet now, then its decaying trail)
uniform vec2 uAnchor;          // lattice origin: follows the magnet with a lag
uniform float uK;              // lattice wave number
uniform float uSigma;
uniform float uPixelRatio;
uniform float uSize;
uniform float uFar;
uniform float uHeight;
varying float vTip;
varying float vAlpha;
varying float vLift;
${NOISE}
float swellAt(vec2 p) {
  // one octave of simplex noise for the swell, a cheap interference ripple for the fine scale
  return 0.15 * snoise(vec3(p * 0.10, uTime * 0.045))
       + 0.035 * sin(p.x * 0.83 + uTime * 0.31) * sin(p.y * 0.71 - uTime * 0.23);
}
float envAt(vec2 p) {
  float env = 0.0;
  for (int i = 0; i < ${TRAIL}; i++) {
    vec2 d = p - uMag[i].xy;
    float s = i == 0 ? uSigma : uSigma * 0.8;
    env = max(env, uMag[i].z * exp(-dot(d, d) / (2.0 * s * s)));
  }
  return env;
}
float fadeAt(float depth) {
  return smoothstep(uFar, uFar * 0.3, depth) * smoothstep(0.6, 2.2, depth);
}
// dots smaller than ~1.6 px fade instead of shrinking, which keeps the dense far field from aliasing into moire
float sizeFade(float px) {
  return clamp((px - 0.35) / 1.4, 0.0, 1.0);
}
`;

// The fluid surface: a grid of dots displaced by a slow swell, lifted under the magnet, with soft bumps where the
// spikes stand.
const VERT = /* glsl */ `
${COMMON}
attribute vec2 aGrid;
void main() {
  vec2 p = aGrid;
  float swell = swellAt(p);
  float env = envAt(p);
  vec2 q = p - uAnchor;
  float L = cos(uK * q.x) + cos(uK * (-0.5 * q.x + 0.8660254 * q.y)) + cos(uK * (-0.5 * q.x - 0.8660254 * q.y));
  L = clamp((L + 1.5) / 4.5, 0.0, 1.0);
  float cone = clamp(1.0 - sqrt(max(0.0, 1.0 - L) / 0.5), 0.0, 1.0);
  float peak = pow(env, 1.4) * cone;
  float h = swell * (1.0 - 0.6 * env) + 0.14 * env + 0.3 * uHeight * peak;
  vec4 mv = modelViewMatrix * vec4(p.x, h, p.y, 1.0);
  gl_Position = projectionMatrix * mv;
  float depth = -mv.z;
  float px = uSize * uPixelRatio / depth;
  gl_PointSize = max(1.0, px);
  vTip = 0.0;
  vLift = clamp(0.58 + swell * 1.6 + peak * 0.6, 0.0, 1.0);
  vAlpha = fadeAt(depth) * sizeFade(px);
}`;

// The spikes: rings of dots on a concave cone at every lattice site near the magnet (the Rosensweig lattice).
// Site positions are the peaks of the three-cosine pattern: anchor + (2 pi / k) (m1 (1, 1/sqrt3) + m2 (0, 2/sqrt3)).
const CONE_VERT = /* glsl */ `
${COMMON}
attribute vec2 aSite;   // lattice indices m1, m2
attribute vec2 aRing;   // height fraction t (0 base .. 1 tip), angle
void main() {
  float a = 6.2831853 / uK;
  vec2 site = uAnchor + a * (aSite.x * vec2(1.0, 0.5773503) + aSite.y * vec2(0.0, 1.1547005));
  float env = envAt(site);
  float hc = uHeight * pow(env, 1.4);
  float t = aRing.x;
  float lattice = 4.0 * 3.1415927 / (1.7320508 * uK);
  float r = 0.3 * lattice * (0.55 + 0.45 * env) * pow(1.0 - t, 1.55);
  float ang = aRing.y + t * 0.5;
  vec2 xz = site + r * vec2(cos(ang), sin(ang));
  float base = swellAt(site) * (1.0 - 0.6 * env) + 0.14 * env;
  vec4 mv = modelViewMatrix * vec4(xz.x, base + t * hc, xz.y, 1.0);
  gl_Position = projectionMatrix * mv;
  float depth = -mv.z;
  float px = 1.45 * uSize * uPixelRatio / depth;
  gl_PointSize = max(1.0, px);
  float shown = smoothstep(0.06, 0.22, hc);
  vTip = smoothstep(0.78, 1.0, t) * smoothstep(0.5, 0.85, hc / uHeight);
  vLift = 0.5 + 0.5 * t;
  vAlpha = fadeAt(depth) * shown * sizeFade(px);
}`;

const FRAG = /* glsl */ `
uniform vec3 uInk;
uniform vec3 uAccent;
varying float vTip;
varying float vAlpha;
varying float vLift;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  float a = smoothstep(0.5, 0.32, r);
  if (a * vAlpha < 0.02) discard;
  vec3 col = mix(uInk * (0.38 + 0.62 * vLift), uAccent, vTip);
  gl_FragColor = vec4(col, a * vAlpha * mix(0.6, 1.0, max(vTip, vLift)));
}`;

/** The fluid surface as a grid of dots on y = 0, far rows first. */
function grid(cols: number, rows: number) {
  const x0 = -12, x1 = 12, zNear = 7.2, zFar = -22;
  const g = new Float32Array(cols * rows * 2);
  let i = 0;
  for (let r = 0; r < rows; r++) {
    const z = zFar + ((zNear - zFar) * r) / (rows - 1);
    for (let c = 0; c < cols; c++) {
      g[i++] = x0 + ((x1 - x0) * c) / (cols - 1);
      g[i++] = z;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("aGrid", new THREE.BufferAttribute(g, 2));
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(cols * rows * 3), 3));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -8), 40);
  return geo;
}

export interface FieldReadout {
  x: number;
  z: number;
  strength: number;
  fps: number;
}

export class Ferrofluid {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(36, 1, 0.1, 80);
  private material: THREE.ShaderMaterial;
  private points: THREE.Points;
  private cones: THREE.Points;
  private coneMaterial: THREE.ShaderMaterial;
  private raycaster = new THREE.Raycaster();
  private plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.25);
  private ndc = new THREE.Vector2();
  private hit = new THREE.Vector3();
  // idle position: mid-depth, right of centre, in the open band between the nav and the hero type
  private target = new THREE.Vector2(1.6, -3.0);
  private mag = new THREE.Vector2(1.6, -3.0);
  private anchor = new THREE.Vector2(1.6, -3.0);
  private strength = 0;
  private speed = 0;
  private trail: THREE.Vector3[] = Array.from({ length: TRAIL - 1 }, () => new THREE.Vector3(0, 0, 0));
  private trailClock = 0;
  private lastPointer = -1e9;
  private raf = 0;
  private running = false;
  private visible = true;
  private last = 0;
  private time = 0;
  private fpsAvg = 60;
  private ro: ResizeObserver;
  private io: IntersectionObserver;
  private reduced: boolean;
  onFrame?: (r: FieldReadout) => void;

  constructor(private host: HTMLElement) {
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(dpr);
    this.renderer.setClearColor(0x000000, 1);
    host.appendChild(this.renderer.domElement);

    // Density adapts to the device: ~140k surface dots + ~20k spike dots on a desktop, ~50k + ~8k on a phone; the
    // frame-time ladder in loop() lowers it further on slow (e.g. integrated) GPUs.
    const small = window.innerWidth < 700; // phones
    const geo = small ? grid(220, 230) : grid(360, 390);

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: true,
      uniforms: {
        uTime: { value: 0 },
        uMag: { value: Array.from({ length: TRAIL }, () => new THREE.Vector3()) },
        uAnchor: { value: this.anchor },
        uK: { value: (4 * Math.PI) / (Math.sqrt(3) * 0.92) }, // lattice constant 0.92
        uSigma: { value: 1.55 },
        uHeight: { value: 1.02 },
        uPixelRatio: { value: dpr },
        uSize: { value: small ? 17 : 14 },
        uFar: { value: 26 },
        uInk: { value: new THREE.Color("#f2f1ec") },
        uAccent: { value: new THREE.Color("#f5b700") },
      },
    });
    if (import.meta.env.DEV) (window as unknown as { __ferrofluid?: Ferrofluid }).__ferrofluid = this;
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // spikes: (2 S + 1)^2 lattice sites around the anchor, rings of dots per site
    const S = small ? 5 : 6;
    const rings = small ? 10 : 13;
    const site: number[] = [];
    const ring: number[] = [];
    for (let m1 = -S; m1 <= S; m1++)
      for (let m2 = -S; m2 <= S; m2++)
        for (let k = 0; k <= rings; k++) {
          const t = k / rings;
          const n = k === rings ? 1 : Math.max(3, Math.round(13 * (1 - t)) + 2);
          for (let j = 0; j < n; j++) {
            site.push(m1, m2);
            ring.push(t, (j / n) * Math.PI * 2 + k * 0.37);
          }
        }
    const cgeo = new THREE.BufferGeometry();
    cgeo.setAttribute("aSite", new THREE.BufferAttribute(new Float32Array(site), 2));
    cgeo.setAttribute("aRing", new THREE.BufferAttribute(new Float32Array(ring), 2));
    cgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((site.length / 2) * 3), 3));
    this.coneMaterial = new THREE.ShaderMaterial({
      vertexShader: CONE_VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: true,
      depthWrite: true,
      uniforms: this.material.uniforms,
    });
    this.cones = new THREE.Points(cgeo, this.coneMaterial);
    this.cones.frustumCulled = false;
    this.scene.add(this.cones);

    this.camera.position.set(0, 2.7, 8.4);
    this.camera.lookAt(0, -0.35, -3);

    this.resize();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.io = new IntersectionObserver(([e]) => {
      this.visible = e.isIntersecting;
      this.sync();
    });
    this.io.observe(host);
    document.addEventListener("visibilitychange", this.sync);
    window.addEventListener("pointermove", this.onPointer, { passive: true });

    if (this.reduced) {
      this.strength = 1;
      this.trail.forEach((t) => t.set(0, 0, 0));
      this.renderFrame(0);
    } else {
      this.sync();
    }
  }

  // adaptive quality: 0 = full, 1 = device-pixel ratio 1, 2 = also a coarser grid
  private quality = 0;
  private samples: number[] = [];

  private adapt(dt: number) {
    if (this.quality >= 2 || this.time < 1.2) return;
    this.samples.push(dt * 1000);
    if (this.samples.length < 90) return;
    const sorted = [...this.samples].sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    this.samples = [];
    if (median < 18.5) {
      this.quality = 3; // fast enough: stop measuring
      return;
    }
    this.quality++;
    if (this.quality === 1) {
      this.renderer.setPixelRatio(1);
      this.material.uniforms.uPixelRatio.value = 1;
      this.resize();
    } else {
      const old = this.points.geometry;
      this.points.geometry = grid(250, 270);
      old.dispose();
    }
  }

  private onPointer = (e: PointerEvent) => {
    const rect = this.host.getBoundingClientRect();
    if (e.clientY > rect.bottom || e.clientY < rect.top) return;
    this.ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    if (this.raycaster.ray.intersectPlane(this.plane, this.hit)) {
      this.target.set(THREE.MathUtils.clamp(this.hit.x, -13, 13), THREE.MathUtils.clamp(this.hit.z, -20, 6.5));
    } else {
      this.target.set(THREE.MathUtils.clamp(this.ndc.x * 9, -13, 13), -20); // above the horizon: the far edge
    }
    this.lastPointer = this.time;
  };

  private sync = () => {
    const should = this.visible && !document.hidden && !this.reduced;
    if (should && !this.running) {
      this.running = true;
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.loop);
    } else if (!should && this.running) {
      this.running = false;
      cancelAnimationFrame(this.raf);
    }
  };

  private resize() {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    // keep the horizon composition on portrait screens
    this.camera.fov = w / h < 1 ? 52 : 36;
    this.camera.updateProjectionMatrix();
    if (!this.running) this.renderFrame(0);
  }

  private loop = (now: number) => {
    if (!this.running) return;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    this.fpsAvg += (1 / Math.max(dt, 1e-3) - this.fpsAvg) * 0.05;
    this.adapt(dt);
    this.renderFrame(dt);
    this.raf = requestAnimationFrame(this.loop);
  };

  private renderFrame(dt: number) {
    this.time += dt;
    const t = this.time;
    // No pointer for a while (or a touch device): the magnet drifts on its own, slowly.
    if (t - this.lastPointer > 5) {
      this.target.set(1.4 + Math.sin(t * 0.13) * 2.6, -3.1 + Math.sin(t * 0.21 + 1.3) * 1.1);
    }
    const prevX = this.mag.x, prevY = this.mag.y;
    const follow = 1 - Math.exp(-dt * 7);
    this.mag.lerp(this.target, follow);
    const v = dt > 0 ? Math.hypot(this.mag.x - prevX, this.mag.y - prevY) / dt : 0;
    this.speed += (v - this.speed) * (1 - Math.exp(-dt * 6));
    // At rest the peaks grow; a fast-moving magnet drags a flatter mound.
    const goal = THREE.MathUtils.clamp(1.05 - this.speed * 0.11, 0.3, 1.0);
    this.strength += (goal - this.strength) * (1 - Math.exp(-dt * (goal > this.strength ? 1.4 : 5)));
    this.anchor.lerp(this.mag, 1 - Math.exp(-dt * 2.2));
    // trail: drop a point every 70 ms, each fading out over ~0.7 s
    this.trailClock += dt;
    if (this.trailClock > 0.07) {
      this.trailClock = 0;
      const p = this.trail.pop()!;
      p.set(this.mag.x, this.mag.y, Math.min(0.85, this.speed * 0.12) * this.strength);
      this.trail.unshift(p);
    }
    const decay = Math.exp(-dt / 0.7);
    for (const p of this.trail) p.z *= decay;
    const u = this.material.uniforms;
    u.uTime.value = t;
    const mags = u.uMag.value as THREE.Vector3[];
    mags[0].set(this.mag.x, this.mag.y, this.strength);
    for (let i = 0; i < this.trail.length; i++) mags[i + 1].copy(this.trail[i]);
    this.renderer.render(this.scene, this.camera);
    this.onFrame?.({ x: this.mag.x, z: this.mag.y, strength: this.strength, fps: this.fpsAvg });
  }

  dispose() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.io.disconnect();
    document.removeEventListener("visibilitychange", this.sync);
    window.removeEventListener("pointermove", this.onPointer);
    this.points.geometry.dispose();
    this.cones.geometry.dispose();
    this.material.dispose();
    this.coneMaterial.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
