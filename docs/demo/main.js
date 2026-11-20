// Interactive demo: MuJoCo (WebAssembly) simulation of a torque-controlled KUKA iiwa 14 with the
// Bregman-CLF controller of controller.js, one arm per selected backstepping metric.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import loadMujoco from "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.js";
import { solveQP } from "https://cdn.jsdelivr.net/npm/quadprog@2.0.0/index.js";
import { BregmanController } from "./controller.js";

const METRICS = {
  hessian: { label: "Hessian", color: 0x1f5fa8 },
  constant: { label: "Constant I", color: 0xc0392b },
  target: { label: "Target Hess.", color: 0xe08a1e },
};
const DT = 1e-3;          // control and physics period [s], as in the paper
const TRAIL = 6000;       // trail points (one per step)
const statusEl = document.getElementById("status");

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function main() {
  const [mujoco, xmlBytes, presetsJson] = await Promise.all([
    loadMujoco(), fetchBytes("model/iiwa14.xml"), fetch("presets.json").then((r) => r.json()),
  ]);
  const { robot, presets } = presetsJson;
  const xml = new TextDecoder().decode(xmlBytes);
  const files = [...xml.matchAll(/<mesh[^>]*file="([^"]+)"/g)].map((m) => m[1]);
  mujoco.FS.mkdir("/model"); mujoco.FS.mkdir("/model/assets");
  mujoco.FS.writeFile("/model/iiwa14.xml", xmlBytes);
  await Promise.all(files.map(async (f) => mujoco.FS.writeFile(`/model/assets/${f}`, await fetchBytes(`model/assets/${f}`))));
  const model = mujoco.MjModel.from_xml_path("/model/iiwa14.xml");
  model.opt.timestep = DT;

  // ---- three.js scene (MuJoCo frame: z up) ----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a909a, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(1.5, -2, 3);
  scene.add(sun);
  const grid = new THREE.GridHelper(4, 40, 0xd0d4da, 0xe9ecf0);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // visual geoms (group <= 2) of the robot: shared geometries, one set of meshes per arm
  const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
  const visual = [];
  const meshGeoms = new Map();
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] > 2 || model.geom_type[g] !== MESH) continue;
    const id = model.geom_dataid[g];
    if (!meshGeoms.has(id)) {
      const va = model.mesh_vertadr[id], vn = model.mesh_vertnum[id], fa = model.mesh_faceadr[id], fn = model.mesh_facenum[id];
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(model.mesh_vert.subarray(3 * va, 3 * (va + vn))), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.subarray(3 * fa, 3 * (fa + fn))), 1));
      geo.computeVertexNormals();
      meshGeoms.set(id, geo);
    }
    const mat = model.geom_matid[g];
    const rgba = mat >= 0 ? model.mat_rgba.subarray(4 * mat, 4 * mat + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    visual.push({ g, geo: meshGeoms.get(id), color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace) });
  }

  const setGroup = new THREE.Group();  // keep-in set and target
  scene.add(setGroup);
  const targetMesh = new THREE.Mesh(new THREE.SphereGeometry(0.014, 24, 16), new THREE.MeshStandardMaterial({ color: 0xf5b800, emissive: 0x6b4f00 }));
  scene.add(targetMesh);

  // ---- one simulation per metric ---------------------------------------------------------------
  class Arm {
    constructor(metric, ghost) {
      this.metric = metric;
      this.data = new mujoco.MjData(model);
      this.ctrl = new BregmanController(mujoco, model, robot, robot.ee_site, solveQP);
      this.ctrl.metric = metric;
      this.group = new THREE.Group();
      const tint = new THREE.Color(METRICS[metric].color);
      this.meshes = visual.map((v) => {
        const m = new THREE.MeshStandardMaterial({
          color: ghost ? v.color.clone().lerp(tint, 0.55) : v.color, roughness: 0.55, metalness: 0.1,
          transparent: ghost, opacity: ghost ? 0.38 : 1, depthWrite: !ghost,
        });
        const mesh = new THREE.Mesh(v.geo, m);
        mesh.matrixAutoUpdate = false;
        this.group.add(mesh);
        return mesh;
      });
      this.trailPos = new Float32Array(3 * TRAIL);
      const tg = new THREE.BufferGeometry();
      tg.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
      this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: METRICS[metric].color, linewidth: 2 }));
      this.trail.frustumCulled = false;
      this.tip = new THREE.Mesh(new THREE.SphereGeometry(0.011, 20, 14), new THREE.MeshStandardMaterial({ color: METRICS[metric].color, emissive: tint.clone().multiplyScalar(0.4) }));
      this.group.add(this.trail, this.tip);
      scene.add(this.group);
    }
    reset(p) {
      mujoco.mj_resetData(model, this.data);
      this.data.qpos.set(p.q0); this.data.qvel.set(p.v0);
      mujoco.mj_forward(model, this.data);
      this.n = 0; this.steps = 0; this.infeasible = 0; this.minMargin = Infinity; this.info = null;
      this.trail.geometry.setDrawRange(0, 0);
    }
    step() {
      mujoco.mj_forward(model, this.data);
      const { tau, info } = this.ctrl.step(this.data);
      for (let i = 0; i < model.nu; i++) this.data.ctrl[i] = Math.max(-robot.tau_max[i], Math.min(robot.tau_max[i], tau[i]));
      mujoco.mj_step(model, this.data);
      this.info = info; this.steps++;
      if (info.fallback) this.infeasible++;
      this.minMargin = Math.min(this.minMargin, info.margin);
      if (this.n < TRAIL) { this.trailPos.set(info.x, 3 * this.n); this.n++; }
    }
    // Add a joint velocity that pushes the end effector toward the nearest face. The speed is what
    // braking at `brake` m/s^2 can still stop before the face (at most `vmax`), so the kick is recoverable.
    kick(brake, vmax) {
      mujoco.mj_forward(model, this.data);
      const c = this.ctrl, x = c.eePos(this.data), l = c.bar.slacks(x), lmin = Math.min(...l);
      if (!(lmin > 0)) return;
      const speed = Math.min(vmax, Math.sqrt(2 * brake * lmin));
      const nrm = c.bar.A[l.indexOf(lmin)];
      mujoco.mj_jacSite(model, this.data, c.jp, c.jr, c.site);
      const J = c.jp.GetView(), nv = model.nv;
      // least-norm dv with J dv = speed * n:  dv = J^T (J J^T)^-1 (speed n)
      const JJt = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) for (let j = 0; j < nv; j++) JJt[3 * r + s] += J[r * nv + j] * J[s * nv + j];
      const y = new THREE.Vector3(...nrm).multiplyScalar(speed).applyMatrix3(new THREE.Matrix3().set(...JJt).invert());
      for (let j = 0; j < nv; j++) this.data.qvel[j] += J[j] * y.x + J[nv + j] * y.y + J[2 * nv + j] * y.z;
    }
    draw() {
      const d = this.data, m4 = new THREE.Matrix4();
      visual.forEach((v, i) => {
        const R = d.geom_xmat.subarray(9 * v.g, 9 * v.g + 9), p = d.geom_xpos.subarray(3 * v.g, 3 * v.g + 3);
        m4.set(R[0], R[1], R[2], p[0], R[3], R[4], R[5], p[1], R[6], R[7], R[8], p[2], 0, 0, 0, 1);
        this.meshes[i].matrix.copy(m4);
      });
      const x = this.ctrl.eePos(d);
      this.tip.position.set(x[0], x[1], x[2]);
      this.trail.geometry.attributes.position.needsUpdate = true;
      this.trail.geometry.setDrawRange(0, this.n);
    }
    dispose() { scene.remove(this.group); this.data.delete(); }
  }

  // ---- state and UI ----------------------------------------------------------------------------
  let arms = [], preset = presets[0], xd = preset.x_d.slice(), time = 0, paused = false, acc = 0, last = performance.now();
  const el = (id) => document.getElementById(id);
  const boxes = [...document.querySelectorAll("input[data-metric]")];

  function buildSet() {
    setGroup.clear();
    const pts = [];
    for (const [a, b] of preset.edges) pts.push(...a, ...b);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    setGroup.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x21b34c })));
  }
  function placeCamera() {
    const A = preset.A, b = preset.b, c = new THREE.Vector3(...preset.center);
    let j = 0, best = Infinity;
    A.forEach((a, i) => { const s = b[i] - (a[0] * xd[0] + a[1] * xd[1] + a[2] * xd[2]); if (s < best) { best = s; j = i; } });
    const az = Math.atan2(A[j][1], A[j][0]) + Math.PI / 2 + Math.PI, elv = 0.32, R = 1.3;
    const look = c.clone().multiplyScalar(0.75).add(new THREE.Vector3(0, 0, 0.1));
    controls.target.copy(look);
    camera.position.set(look.x + R * Math.cos(az) * Math.cos(elv), look.y + R * Math.sin(az) * Math.cos(elv), look.z + R * Math.sin(elv));
  }
  function applyTarget() {
    targetMesh.position.set(xd[0], xd[1], xd[2]);
    for (const a of arms) a.ctrl.setScene(preset.A, preset.b, xd);
  }
  function rebuildArms() {
    const want = boxes.filter((b) => b.checked).map((b) => b.dataset.metric);
    arms.forEach((a) => a.dispose());
    arms = want.map((m, i) => new Arm(m, i > 0));
    restart();
  }
  function restart() {
    xd = preset.x_d.slice(); time = 0; acc = 0;
    applyTarget();
    arms.forEach((a) => a.reset(preset));
  }
  function newTarget() {  // 2 to 10 mm inside a random face, as in the paper's near-boundary condition
    const A = preset.A, b = preset.b, slack = (x) => Math.min(...A.map((a, i) => b[i] - (a[0] * x[0] + a[1] * x[1] + a[2] * x[2])));
    for (let tries = 0; tries < 2000; tries++) {
      const u = new THREE.Vector3().randomDirection(), c = preset.center;
      let t = Infinity;  // exit distance from the centre along u
      A.forEach((a, i) => { const den = a[0] * u.x + a[1] * u.y + a[2] * u.z; if (den > 1e-9) t = Math.min(t, (b[i] - (a[0] * c[0] + a[1] * c[1] + a[2] * c[2])) / den); });
      const depth = 0.002 + 0.008 * Math.random();
      const x = [c[0] + (t - depth * 1.5) * u.x, c[1] + (t - depth * 1.5) * u.y, c[2] + (t - depth * 1.5) * u.z];
      const s = slack(x);
      if (s > 0.002 && s < 0.012 && x[2] > 0.15) { xd = x; applyTarget(); return; }
    }
  }
  presets.forEach((p, i) => el("preset").add(new Option(`Start ${String.fromCharCode(65 + i)} (seed ${p.seed})`, i)));
  el("preset").onchange = (e) => { preset = presets[+e.target.value]; buildSet(); restart(); placeCamera(); };
  boxes.forEach((b) => (b.onchange = () => { if (!boxes.some((x) => x.checked)) b.checked = true; rebuildArms(); }));
  el("restart").onclick = restart;
  el("pause").onclick = () => { paused = !paused; el("pause").textContent = paused ? "Play" : "Pause"; };
  el("kick").onclick = () => arms.forEach((a) => a.kick(20, 0.5));
  el("target").onclick = newTarget;
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "r" || e.key === "R") restart();
    else if (e.key === " ") { e.preventDefault(); el("pause").click(); }
    else if (e.key === "k" || e.key === "K") el("kick").click();
    else if (e.key === "t" || e.key === "T") newTarget();
  });
  function resize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);

  function readout() {
    el("time").textContent = `t = ${time.toFixed(2)} s`;
    el("readout").innerHTML = arms.map((a) => {
      const i = a.info, out = a.minMargin < 0;
      const margin = !i ? "" : out ? `out ${(-1e3 * a.minMargin).toFixed(0)} mm` : i.margin < 1e-4 ? `${(1e6 * i.margin).toFixed(0)} µm` : `${(1e3 * i.margin).toFixed(1)} mm`;
      const need = !i || !Number.isFinite(i.demand) ? "–" : i.demand >= 1e4 ? i.demand.toExponential(0) : i.demand.toFixed(1);
      const inf = a.steps ? `${(100 * a.infeasible / a.steps).toFixed(0)} %` : "";
      return `<tr><td><span class="swatch" style="display:inline-block;background:#${METRICS[a.metric].color.toString(16).padStart(6, "0")}"></span> ${METRICS[a.metric].label}</td>`
        + `<td class="${out ? "out" : ""}">${margin}</td><td>${need}</td><td>${inf}</td></tr>`;
    }).join("");
  }

  function frame(now) {
    const dtFrame = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (!paused) {
      acc += dtFrame * parseFloat(el("speed").value);
      let n = Math.min(Math.floor(acc / DT), 60);
      acc -= n * DT;
      for (; n > 0; n--) { arms.forEach((a) => a.step()); time += DT; }
    }
    arms.forEach((a) => a.draw());
    readout();
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  buildSet(); rebuildArms(); placeCamera(); resize();
  const pre = parseInt(new URLSearchParams(location.search).get("steps") || "0", 10);  // ?steps=N: advance N steps before the first frame
  for (let i = 0; i < pre; i++) { arms.forEach((a) => a.step()); time += DT; }
  if (pre) { paused = true; el("pause").textContent = "Play"; console.log("demo-state " + JSON.stringify(arms.map((a) => ({ metric: a.metric, minMarginMm: +(1e3 * a.minMargin).toFixed(2), infeasible: +(a.infeasible / a.steps).toFixed(3) })))); }
  el("panel").hidden = false;
  statusEl.remove();
  window.__demo = { arms: () => arms, time: () => time };  // for testing
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `The demo could not start: ${err.message}. It needs a recent browser with WebAssembly and WebGL.`;
});
