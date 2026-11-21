// Line-by-line port of picovla/sim.py. Same operation order, so the physics
// matches Python to the last bit (tests/test_parity.py checks this).
export const COLORS = { red: [220, 50, 47], green: [80, 170, 60], blue: [40, 110, 220], yellow: [235, 190, 30] };
const BG = [238, 232, 220], ARM = [90, 90, 100], TIP = [30, 30, 35];
export const BASE = [0.0, -1.25];
const L1 = 1.2, L2 = 1.05, RB = 0.09, RT = 0.05, SPEED = 0.04, SUB = 4;
export const MAX_STEPS = 120;
const PUSH_DIST = 0.25;
export const DIRS = { left: [-1, 0], right: [1, 0], up: [0, 1], down: [0, -1] };

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function reachable(x, y) {
  const dx = x - BASE[0], dy = y - BASE[1];
  const d = Math.sqrt(dx * dx + dy * dy);
  return 0.25 < d && d < L1 + L2 - 0.02;
}

export class Env {
  constructor(blocks, tip, task) {
    this.blocks = {};
    for (const c of Object.keys(blocks)) this.blocks[c] = [blocks[c][0], blocks[c][1]];
    this.tip = [tip[0], tip[1]];
    this.task = task; // {kind, color, arg}
    this.start = {};
    for (const c of Object.keys(blocks)) this.start[c] = [blocks[c][0], blocks[c][1]];
    this.t = 0;
  }

  static random(rand = Math.random) {
    const names = Object.keys(COLORS).sort(() => rand() - 0.5).slice(0, 3);
    const blocks = {};
    let n = 0;
    while (n < 3) {
      const x = -0.6 + 1.2 * rand(), y = -0.35 + 0.85 * rand();
      if (Object.values(blocks).every(([a, b]) => (x - a) ** 2 + (y - b) ** 2 > 0.32 ** 2)) blocks[names[n++]] = [x, y];
    }
    let tip;
    do tip = [-0.6 + 1.2 * rand(), -0.75 + 0.3 * rand()];
    while (!Object.values(blocks).every(([a, b]) => (tip[0] - a) ** 2 + (tip[1] - b) ** 2 > 0.25 ** 2));
    return new Env(blocks, tip, null);
  }

  step(action) {
    let ax = action[0], ay = action[1];
    const n = Math.sqrt(ax * ax + ay * ay);
    if (n > 1.0) { ax = ax / n; ay = ay / n; }
    for (let s = 0; s < SUB; s++) {
      const nx = clamp(this.tip[0] + ax * SPEED / SUB, -1.0, 1.0);
      const ny = clamp(this.tip[1] + ay * SPEED / SUB, -1.0, 1.0);
      if (reachable(nx, ny)) this.tip = [nx, ny];
      this.resolve();
    }
    this.t += 1;
    return this.task ? this.success() : false;
  }

  resolve() {
    const names = Object.keys(this.blocks).sort();
    for (let it = 0; it < 3; it++) {
      for (const c of names) {
        const b = this.blocks[c];
        const dx = b[0] - this.tip[0], dy = b[1] - this.tip[1];
        const d = Math.sqrt(dx * dx + dy * dy);
        if (1e-9 < d && d < RB + RT) {
          const k = (RB + RT - d) / d;
          b[0] += dx * k;
          b[1] += dy * k;
        }
      }
      for (let i = 0; i < names.length; i++) {
        for (let j = i + 1; j < names.length; j++) {
          const a = this.blocks[names[i]], b = this.blocks[names[j]];
          const dx = b[0] - a[0], dy = b[1] - a[1];
          const d = Math.sqrt(dx * dx + dy * dy);
          if (1e-9 < d && d < 2 * RB) {
            const k = (2 * RB - d) / d * 0.5;
            a[0] -= dx * k; a[1] -= dy * k;
            b[0] += dx * k; b[1] += dy * k;
          }
        }
      }
      for (const b of Object.values(this.blocks)) {
        b[0] = clamp(b[0], -1.0 + RB, 1.0 - RB);
        b[1] = clamp(b[1], -1.0 + RB, 1.0 - RB);
      }
    }
  }

  success() {
    const t = this.task, b = this.blocks[t.color];
    if (!b) return false;
    if (t.kind === "touch") {
      const dx = b[0] - this.tip[0], dy = b[1] - this.tip[1];
      return Math.sqrt(dx * dx + dy * dy) < RB + RT + 0.015;
    }
    const s = this.start[t.color];
    if (t.kind === "push") {
      const [ux, uy] = DIRS[t.arg];
      const dx = b[0] - s[0], dy = b[1] - s[1];
      return dx * ux + dy * uy >= PUSH_DIST && Math.abs(-dx * uy + dy * ux) < 0.12;
    }
    const o = this.blocks[t.arg];
    if (!o) return false;
    const dx = b[0] - o[0], dy = b[1] - o[1];
    return Math.sqrt(dx * dx + dy * dy) < 2 * RB + 0.05;
  }
}

export function elbow(tip) {
  const px = tip[0] - BASE[0], py = tip[1] - BASE[1];
  const d = clamp(Math.sqrt(px * px + py * py), 1e-6, L1 + L2);
  const c = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1.0, 1.0);
  const th = Math.atan2(py, px) - Math.acos(c);
  return [BASE[0] + L1 * Math.cos(th), BASE[1] + L1 * Math.sin(th)];
}

function segDist2(px, py, a, b) {
  const vx = b[0] - a[0], vy = b[1] - a[1];
  const wx = px - a[0], wy = py - a[1];
  const t = clamp((wx * vx + wy * vy) / (vx * vx + vy * vy), 0.0, 1.0);
  const ex = wx - t * vx, ey = wy - t * vy;
  return ex * ex + ey * ey;
}

// RGB bytes, row-major, row 0 is y = +1. Same pixel test as the numpy version.
export function render(env, size = 64, out = new Uint8Array(size * size * 3)) {
  const e = elbow(env.tip);
  const names = Object.keys(env.blocks).sort();
  for (let r = 0; r < size; r++) {
    const gy = -((r + 0.5) / size * 2.0 - 1.0);
    for (let col = 0; col < size; col++) {
      const gx = (col + 0.5) / size * 2.0 - 1.0;
      let px = BG;
      for (const c of names) {
        const b = env.blocks[c];
        if ((gx - b[0]) * (gx - b[0]) + (gy - b[1]) * (gy - b[1]) < RB * RB) px = COLORS[c];
      }
      if (segDist2(gx, gy, BASE, e) < 0.045 * 0.045) px = ARM;
      if (segDist2(gx, gy, e, env.tip) < 0.035 * 0.035) px = ARM;
      if ((gx - env.tip[0]) * (gx - env.tip[0]) + (gy - env.tip[1]) * (gy - env.tip[1]) < RT * RT) px = TIP;
      const o = (r * size + col) * 3;
      out[o] = px[0]; out[o + 1] = px[1]; out[o + 2] = px[2];
    }
  }
  return out;
}
