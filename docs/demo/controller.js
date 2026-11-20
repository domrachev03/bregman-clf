// Bregman-CLF inverse-dynamics controller, JavaScript port for the browser demo.
//
// Task: end-effector position x in a keep-in polytope {A x < b}. Generating function
//   psi(x) = P/2 |x|^2 - mu * sum_i log(b_i - a_i^T x),
// Lyapunov function V2 = D_psi(x_d, x) + 1/2 s^T Gamma s with s = x_dot - x_dot_r and
//   x_dot_r = -k H(x)^-1 (grad psi(x) - grad psi(x_d)),  H = Hess psi.
// Metric Gamma: "hessian" (H(x), proposed), "constant" (I) or "target" (H(x_d)).
//
// Each control step solves the QP over the joint acceleration vdot:
//   min  1/2 |J vdot + Jdot v - a*|^2 + 1e-3/2 |vdot - ((q_home - q) - v)|^2 + 1e-6/2 |vdot|^2
//   s.t. V2_dot + gamma V2 <= 0                    (rate constraint, one row)
//        |M vdot + h| <= tau_max                   (torque bounds)
//        joint-velocity and joint-position rows
// and applies tau = M vdot + h. If the QP is infeasible or x is outside the set, it applies the
// fallback tau = g(q) - 15 M(q) v. Gains as in the paper: k = ks = 5, mu = 1e-3, P = 1e-2.
//
// Differences from the Python implementation of the paper: the dynamics terms come from MuJoCo
// instead of Pinocchio, the QP solver is quadprog (Goldfarb-Idnani) instead of DAQP, and the
// joint-position rows use a plain braking-distance velocity cap instead of the calibrated
// braking-aware certificate.

export const GAINS = { k: 5.0, ks: 5.0, mu: 1e-3, P: 1e-2, postureWeight: 1e-3, reg: 1e-6, fallbackKd: 15.0,
  velRecovery: 20.0, jointBrake: 8.0, jointMargin: 0.005 };

// ---- small dense helpers (3-vectors, 3x3 row-major) -------------------------------------------
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mv3 = (M, v) => [M[0] * v[0] + M[1] * v[1] + M[2] * v[2], M[3] * v[0] + M[4] * v[1] + M[5] * v[2], M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];
function solve3(M, r) {  // M x = r for symmetric positive definite M (Cramer)
  const [a, b, c, d, e, f, g, h, i] = M;
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    (r[0] * A + r[1] * (c * h - b * i) + r[2] * (b * f - c * e)) / det,
    (r[0] * B + r[1] * (a * i - c * g) + r[2] * (c * d - a * f)) / det,
    (r[0] * C + r[1] * (b * g - a * h) + r[2] * (a * e - b * d)) / det,
  ];
}

// ---- generating function ---------------------------------------------------------------------
export class Barrier {
  constructor(A, b, mu = GAINS.mu, P = GAINS.P) { this.A = A; this.b = b; this.mu = mu; this.P = P; }
  slacks(x) { return this.A.map((a, i) => this.b[i] - dot3(a, x)); }
  margin(x) { return Math.min(...this.slacks(x)); }
  psi(x) { let s = 0; for (const l of this.slacks(x)) s += Math.log(l); return 0.5 * this.P * dot3(x, x) - this.mu * s; }
  grad(x) {
    const g = [this.P * x[0], this.P * x[1], this.P * x[2]], l = this.slacks(x);
    this.A.forEach((a, i) => { const w = this.mu / l[i]; g[0] += w * a[0]; g[1] += w * a[1]; g[2] += w * a[2]; });
    return g;
  }
  _weighted(w) {  // P I * (p ? 1 : 0) handled by caller; returns sum_i w_i a_i a_i^T
    const H = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    this.A.forEach((a, i) => { for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[3 * r + c] += w[i] * a[r] * a[c]; });
    return H;
  }
  hess(x) {
    const l = this.slacks(x), H = this._weighted(l.map((li) => this.mu / (li * li)));
    H[0] += this.P; H[4] += this.P; H[8] += this.P;
    return H;
  }
  hessDot(x, xdot) {  // d/dt Hess psi(x) along xdot
    const l = this.slacks(x);
    return this._weighted(this.A.map((a, i) => (2 * this.mu * dot3(a, xdot)) / (l[i] * l[i] * l[i])));
  }
  bregman(xd, x) {  // D_psi(x_d, x)
    const g = this.grad(x);
    return this.psi(xd) - this.psi(x) - (g[0] * (xd[0] - x[0]) + g[1] * (xd[1] - x[1]) + g[2] * (xd[2] - x[2]));
  }
}

// Rate constraint  c^T a <= r  on the task acceleration a, the backstepping law a*, and monitors.
export function bregmanRow(bar, metric, x, xdot, xd, k = GAINS.k, ks = GAINS.ks) {
  const gamma = Math.min(k, 2 * ks);
  const H = bar.hess(x), Hdot = bar.hessDot(x, xdot);
  const g = bar.grad(x), gd = bar.grad(xd);
  const e = solve3(H, [g[0] - gd[0], g[1] - gd[1], g[2] - gd[2]]);
  const xdotR = [-k * e[0], -k * e[1], -k * e[2]];
  const He = solve3(H, mv3(Hdot, e));
  const xddotR = [-k * (xdot[0] - He[0]), -k * (xdot[1] - He[1]), -k * (xdot[2] - He[2])];
  const s = [xdot[0] - xdotR[0], xdot[1] - xdotR[1], xdot[2] - xdotR[2]];
  const dx = [x[0] - xd[0], x[1] - xd[1], x[2] - xd[2]];
  const Gamma = metric === "hessian" ? H : metric === "target" ? bar.hess(xd) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const Gs = mv3(Gamma, s);
  const W = bar.bregman(xd, x);
  const V2 = W + 0.5 * dot3(s, Gs);
  let r = -gamma * V2 - dot3(dx, mv3(H, xdot)) + dot3(Gs, xddotR);
  let sd;
  if (metric === "hessian") {
    const Hds = mv3(Hdot, s);
    r -= 0.5 * dot3(s, Hds);
    const t = solve3(H, Hds);
    sd = [-dx[0] - ks * s[0] - 0.5 * t[0], -dx[1] - ks * s[1] - 0.5 * t[1], -dx[2] - ks * s[2] - 0.5 * t[2]];
  } else {
    const t = solve3(Gamma, mv3(H, dx));
    sd = [-t[0] - ks * s[0], -t[1] - ks * s[1], -t[2] - ks * s[2]];
  }
  const aStar = [xddotR[0] + sd[0], xddotR[1] + sd[1], xddotR[2] + sd[2]];
  const nc = Math.hypot(Gs[0], Gs[1], Gs[2]);
  const demand = r >= 0 ? 0 : nc === 0 ? Infinity : -r / nc;  // least |a| that satisfies the row
  return { c: Gs, r, aStar, V2, W, demand };
}

// ---- controller ------------------------------------------------------------------------------
export class BregmanController {
  // mujoco: loaded module; model/data: MjModel/MjData; robot: {tau_max, v_max, q_min, q_max, home}
  constructor(mujoco, model, robot, siteName, solveQP) {
    this.mj = mujoco; this.model = model; this.robot = robot; this.solveQP = solveQP;
    this.nv = model.nv;
    this.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, siteName);
    this.body = model.site_bodyid[this.site];
    this.jp = new mujoco.DoubleBuffer(3 * this.nv); this.jr = new mujoco.DoubleBuffer(3 * this.nv);
    this.jd = new mujoco.DoubleBuffer(3 * this.nv); this.jdr = new mujoco.DoubleBuffer(3 * this.nv);
    this.Mb = new mujoco.DoubleBuffer(this.nv * this.nv);
    this.scratch = new mujoco.MjData(model);  // gravity term of the fallback law
    this.metric = "hessian";
    this.setScene([[1, 0, 0]], [10], [0, 0, 0]);
  }
  setScene(A, b, xd) { this.bar = new Barrier(A, b); this.xd = xd.slice(); }
  eePos(data) { return Array.from(data.site_xpos.subarray(3 * this.site, 3 * this.site + 3)); }

  // data must be up to date at (qpos, qvel) (mj_forward). Returns {tau, info}.
  step(data) {
    const mj = this.mj, model = this.model, nv = this.nv, G = GAINS, dt = model.opt.timestep;
    const q = data.qpos, v = data.qvel;
    const x = this.eePos(data);
    mj.mj_jacSite(model, data, this.jp, this.jr, this.site);
    mj.mj_jacDot(model, data, this.jd, this.jdr, x, this.body);
    mj.mj_fullM(model, data, this.Mb);
    const J = this.jp.GetView(), Jd = this.jd.GetView(), M = this.Mb.GetView();
    const h = new Float64Array(nv);
    for (let i = 0; i < nv; i++) h[i] = data.qfrc_bias[i] - data.qfrc_passive[i];
    const xdot = [0, 0, 0], Jdv = [0, 0, 0];
    for (let r = 0; r < 3; r++) for (let j = 0; j < nv; j++) { xdot[r] += J[r * nv + j] * v[j]; Jdv[r] += Jd[r * nv + j] * v[j]; }

    const margin = this.bar.margin(x);
    const info = { x, margin, fallback: false, reason: "", V2: NaN, demand: NaN, aStar: NaN };
    if (!(margin > 0) || !(this.bar.margin(this.xd) > 0)) {
      info.fallback = true; info.reason = "outside the set";
      return { tau: this.fallback(data, M), info };
    }
    const row = bregmanRow(this.bar, this.metric, x, xdot, this.xd);
    info.V2 = row.V2; info.demand = row.demand; info.aStar = Math.hypot(...row.aStar);

    // cost: D = J^T J + (w_p + reg) I ; d = J^T (a* - Jdot v) + w_p ((q_home - q) - v)
    const D = [null], d = [0];
    const res = [row.aStar[0] - Jdv[0], row.aStar[1] - Jdv[1], row.aStar[2] - Jdv[2]];
    for (let i = 0; i < nv; i++) {
      const Di = [0];
      for (let j = 0; j < nv; j++) {
        Di.push(J[i] * J[j] + J[nv + i] * J[nv + j] + J[2 * nv + i] * J[2 * nv + j] + (i === j ? G.postureWeight + G.reg + 1e-12 : 0));
      }
      D.push(Di);
      d.push(J[i] * res[0] + J[nv + i] * res[1] + J[2 * nv + i] * res[2] + G.postureWeight * ((this.robot.home[i] - q[i]) - v[i]));
    }

    // constraints  a^T vdot >= b0, rows scaled to unit norm
    const rows = [], rhs = [];
    const add = (a, b0) => { const n = Math.max(Math.hypot(...a), 1e-9); rows.push(a.map((ai) => ai / n)); rhs.push(b0 / n); };
    const cJ = new Array(nv);
    for (let j = 0; j < nv; j++) cJ[j] = -(row.c[0] * J[j] + row.c[1] * J[nv + j] + row.c[2] * J[2 * nv + j]);
    add(cJ, -(row.r - dot3(row.c, Jdv)));                                    // rate constraint
    for (let i = 0; i < nv; i++) {                                           // torque bounds
      const Mi = Array.from(M.subarray(i * nv, (i + 1) * nv)), tm = this.robot.tau_max[i];
      add(Mi, -tm - h[i]); add(Mi.map((m) => -m), h[i] - tm);
    }
    for (let i = 0; i < nv; i++) {                                           // joint velocity and position
      const vm = this.robot.v_max[i], step = dt * G.velRecovery;
      const capHi = Math.sqrt(2 * G.jointBrake * Math.max(this.robot.q_max[i] - G.jointMargin - q[i], 0));
      const capLo = Math.sqrt(2 * G.jointBrake * Math.max(q[i] - this.robot.q_min[i] - G.jointMargin, 0));
      const ub = Math.max(Math.min(vm, capHi), v[i] - step), lb = Math.min(Math.max(-vm, -capLo), v[i] + step);
      const e = new Array(nv).fill(0); e[i] = 1;
      add(e, (lb - v[i]) / dt); add(e.map((z) => -z), -(ub - v[i]) / dt);
    }
    const Amat = [null];
    for (let i = 0; i < nv; i++) { const Ai = [0]; for (let c = 0; c < rows.length; c++) Ai.push(rows[c][i]); Amat.push(Ai); }
    const sol = this.solveQP(D, d, Amat, [0, ...rhs]);
    let ok = sol.message === "" && sol.solution;
    const vdot = new Float64Array(nv);
    if (ok) for (let i = 0; i < nv; i++) { vdot[i] = sol.solution[i + 1]; if (!Number.isFinite(vdot[i])) ok = false; }
    if (!ok) {
      info.fallback = true; info.reason = "QP infeasible";
      return { tau: this.fallback(data, M), info };
    }
    const tau = new Float64Array(nv);
    for (let i = 0; i < nv; i++) { let t = h[i]; for (let j = 0; j < nv; j++) t += M[i * nv + j] * vdot[j]; tau[i] = t; }
    return { tau, info };
  }

  fallback(data, M) {  // tau = g(q) - kd M v, clipped
    const mj = this.mj, nv = this.nv, s = this.scratch;
    s.qpos.set(data.qpos); s.qvel.fill(0);
    mj.mj_forward(this.model, s);
    const tau = new Float64Array(nv);
    for (let i = 0; i < nv; i++) {
      let t = s.qfrc_bias[i];
      for (let j = 0; j < nv; j++) t -= GAINS.fallbackKd * M[i * nv + j] * data.qvel[j];
      tau[i] = Math.max(-this.robot.tau_max[i], Math.min(this.robot.tau_max[i], t));
    }
    return tau;
  }
}

// One control + physics step: forward, control, clip to the actuator range, step.
export function simStep(mujoco, model, data, ctrl) {
  mujoco.mj_forward(model, data);
  const { tau, info } = ctrl.step(data);
  for (let i = 0; i < model.nu; i++) data.ctrl[i] = Math.max(-ctrl.robot.tau_max[i], Math.min(ctrl.robot.tau_max[i], tau[i]));
  mujoco.mj_step(model, data);
  return info;
}
