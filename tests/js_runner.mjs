// Reads cases from stdin, replays them with web/sim.js (and web/model.js), prints JSON.
import { readFileSync } from "node:fs";
import { Env, render } from "../web/sim.js";
import { loadModel, predict, tokenize } from "../web/model.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const out = { sim: [], model: [] };
for (const c of input.sim) {
  const env = new Env(c.blocks, c.tip, c.task);
  const succ = c.actions.map((a) => env.step(a));
  out.sim.push({ blocks: env.blocks, tip: env.tip, succ, img: Array.from(render(env)) });
}
if (input.model) {
  const fileFetch = async (p) => ({ json: async () => JSON.parse(readFileSync(p, "utf8")), arrayBuffer: async () => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); } });
  const m = await loadModel(input.model.path, fileFetch);
  for (const c of input.model.cases) {
    out.model.push({ tok: tokenize(m.meta, c.text), act: predict(m, Uint8Array.from(c.img), tokenize(m.meta, c.text), c.state) });
  }
}
process.stdout.write(JSON.stringify(out));
