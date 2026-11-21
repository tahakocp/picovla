// Loads the demo in headless Chrome, runs a few instructions through the UI, reports success and speed.
import puppeteer from "puppeteer-core";
import { serve } from "./serve.mjs";

const chrome = process.env.CHROME || "/usr/bin/google-chrome";
const srv = await serve();
const url = `http://127.0.0.1:${srv.address().port}/`;
const browser = await puppeteer.launch({ executablePath: chrome, args: ["--no-sandbox"] });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(url);
await page.waitForFunction(() => window.__picovla?.ready, { timeout: 30000 });

const N = Number(process.env.N || 12);
let ok = 0, ms = [];
for (let i = 0; i < N; i++) {
  const r = await page.evaluate(async (i) => {
    const p = window.__picovla;
    p.newScene();
    const c = Object.keys(p.env.blocks);
    const texts = [`push the ${c[0]} block left`, `touch the ${c[1]} block`, `move ${c[2]} next to ${c[0]}`, `slide ${c[1]} up`];
    document.getElementById("text").value = texts[i % 4];
    for (let t = 0; t < 120; t++) if (p.stepOnce()) return { done: true, t: p.env.t, text: texts[i % 4] };
    return { done: false, t: p.env.t, text: texts[i % 4] };
  }, i);
  const inf = await page.$eval("#status", (e) => Number(e.textContent.match(/inference (\d+)/)[1]));
  ms.push(inf);
  if (r.done) ok++;
  console.log(`${r.done ? "done" : "fail"}  ${String(r.t).padStart(3)} steps  ${inf} ms  "${r.text}"`);
}
await browser.close();
srv.close();
console.log(`${ok}/${N} done, median inference ${ms.sort((a, b) => a - b)[ms.length >> 1]} ms`);
if (errors.length) { console.error(errors); process.exit(1); }
if (ok < N / 2) process.exit(1);
