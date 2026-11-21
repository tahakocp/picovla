import { Env, render, MAX_STEPS, COLORS, DIRS } from "./sim.js";
import { loadModel, predict, tokenize } from "./model.js";

const $ = (id) => document.getElementById(id);
const cv = $("view"), ctx = cv.getContext("2d");
const big = 448, small = 64;
const EXEC = 4; // actions executed from each predicted chunk of 8, same as the README numbers
let model, env, plan = [], planAt = 0, running = false, lastMs = 0;

// Only used to show a success tick; the policy never sees this parse.
export function parseTask(text, colors) {
  const w = text.toLowerCase().match(/[a-z]+/g) || [];
  const cs = w.filter((x) => x in COLORS && colors.includes(x));
  const dir = w.find((x) => x in DIRS);
  if (!cs.length) return null;
  if (cs.length >= 2) return { kind: "toward", color: cs[0], arg: cs[1] };
  if (dir) return { kind: "push", color: cs[0], arg: dir };
  return { kind: "touch", color: cs[0], arg: null };
}

function newScene() {
  env = Env.random();
  plan = [];
  planAt = 0;
  draw();
  status();
}

function draw() {
  const img = render(env, big);
  const id = ctx.createImageData(big, big);
  for (let i = 0, j = 0; i < big * big; i++) { id.data[j++] = img[3 * i]; id.data[j++] = img[3 * i + 1]; id.data[j++] = img[3 * i + 2]; id.data[j++] = 255; }
  ctx.putImageData(id, 0, 0);
  // what the model actually sees
  const s = render(env, small), sid = ctx.createImageData(small, small);
  for (let i = 0, j = 0; i < small * small; i++) { sid.data[j++] = s[3 * i]; sid.data[j++] = s[3 * i + 1]; sid.data[j++] = s[3 * i + 2]; sid.data[j++] = 255; }
  ctx.putImageData(sid, big - small - 6, big - small - 6);
  ctx.strokeStyle = "#0006";
  ctx.strokeRect(big - small - 6.5, big - small - 6.5, small + 1, small + 1);
  if ($("showplan").checked && plan.length) {
    let [x, y] = env.tip;
    ctx.fillStyle = "#1f5fbfcc";
    for (let k = planAt; k < plan.length; k++) {
      let [ax, ay] = plan[k];
      const n = Math.hypot(ax, ay);
      if (n > 1) { ax /= n; ay /= n; }
      x += ax * 0.04; y += ay * 0.04;
      ctx.beginPath();
      ctx.arc((x + 1) / 2 * big, (1 - y) / 2 * big, 3, 0, 7);
      ctx.fill();
    }
  }
}

function status(extra = "") {
  const t = parseTask($("text").value, Object.keys(env.blocks));
  env.task = t;
  const ok = t && env.success();
  const tok = model ? tokenize(model.meta, $("text").value).filter((i) => i > 0).map((i) => model.meta.vocab[i]).join(" ") : "";
  $("status").innerHTML = `step ${env.t}/${MAX_STEPS}   inference ${lastMs.toFixed(0)} ms\n` +
    `model reads: ${tok.replace(/<unk>/g, "?")}\n` +
    (t ? `task: ${t.kind} ${t.color}${t.arg ? " " + t.arg : ""}  ${ok ? '<span class="ok">done</span>' : ""}` : "task: (no colour in this scene)") +
    (extra ? "\n" + extra : "");
  return ok;
}

function stepOnce() {
  if (!plan.length || planAt >= EXEC) {
    const t0 = performance.now();
    plan = predict(model, render(env, small), tokenize(model.meta, $("text").value), env.tip);
    lastMs = performance.now() - t0;
    planAt = 0;
  }
  env.step(plan[planAt++]);
  draw();
  return status();
}

function loop() {
  if (!running) return;
  const ok = stepOnce();
  if (ok || env.t >= MAX_STEPS) { running = false; $("run").textContent = "Run"; return; }
  setTimeout(loop, 60);
}

$("run").onclick = () => {
  running = !running;
  $("run").textContent = running ? "Stop" : "Run";
  if (running) { if (env.t >= MAX_STEPS) newScene(); loop(); }
};
$("step").onclick = () => { running = false; $("run").textContent = "Run"; stepOnce(); };
$("scene").onclick = () => { running = false; $("run").textContent = "Run"; newScene(); };
$("text").oninput = () => { plan = []; status(); };
$("showplan").onchange = draw;

function chips() {
  const c = Object.keys(env.blocks);
  const ex = [`push the ${c[0]} block left`, `touch the ${c[1]} block`, `move ${c[2]} next to ${c[0]}`, `slide ${c[1]} up`, `shove the ${c[2]} block down`];
  $("chips").innerHTML = "";
  for (const s of ex) {
    const b = document.createElement("button");
    b.textContent = s;
    b.onclick = () => { $("text").value = s; plan = []; status(); };
    $("chips").appendChild(b);
  }
}

newScene();
chips();
$("scene").addEventListener("click", chips);
model = await loadModel("model");
$("text").value = `push the ${Object.keys(env.blocks)[0]} block left`;
status("model loaded. Press Run.");
window.__picovla = { get env() { return env; }, stepOnce, newScene, ready: true };
