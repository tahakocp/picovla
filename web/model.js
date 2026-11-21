// The PicoVLA forward pass in plain JavaScript, no ML library.
// Mirrors picovla/model.py; tests/test_parity.py compares the outputs.

export async function loadModel(base, fetchFn = fetch) {
  const meta = await (await fetchFn(base + ".json")).json();
  const buf = await (await fetchFn(base + ".bin")).arrayBuffer();
  const all = new Float32Array(buf);
  const W = {};
  for (const [k, t] of Object.entries(meta.tensors)) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    W[k] = all.subarray(t.offset, t.offset + n);
  }
  return { meta, W, d: meta.tensors["img_pos"].shape[1], layers: Object.keys(meta.tensors).filter((k) => /^blocks\.\d+\.ln1\.weight$/.test(k)).length };
}

export function tokenize(meta, text) {
  const words = (text.toLowerCase().match(/[a-z]+/g) || []).slice(0, meta.max_len);
  const ids = words.map((w) => { const i = meta.vocab.indexOf(w); return i < 0 ? 1 : i; });
  while (ids.length < meta.max_len) ids.push(0);
  return ids;
}

const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

function conv(x, C, S, w, b, O, k, stride, pad) {
  const So = Math.floor((S + 2 * pad - k) / stride) + 1;
  const y = new Float32Array(O * So * So);
  for (let o = 0; o < O; o++) {
    for (let i = 0; i < So; i++) {
      for (let j = 0; j < So; j++) {
        let s = b[o];
        for (let c = 0; c < C; c++) {
          for (let ki = 0; ki < k; ki++) {
            const r = i * stride + ki - pad;
            if (r < 0 || r >= S) continue;
            for (let kj = 0; kj < k; kj++) {
              const q = j * stride + kj - pad;
              if (q < 0 || q >= S) continue;
              s += w[((o * C + c) * k + ki) * k + kj] * x[(c * S + r) * S + q];
            }
          }
        }
        y[(o * So + i) * So + j] = gelu(s);
      }
    }
  }
  return [y, So];
}

// x: T rows of n -> T rows of m, w is (m, n) like torch.nn.Linear
function linear(x, T, n, w, b, m) {
  const y = new Float32Array(T * m);
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < m; o++) {
      let s = b ? b[o] : 0;
      const wo = o * n, xo = t * n;
      for (let i = 0; i < n; i++) s += w[wo + i] * x[xo + i];
      y[t * m + o] = s;
    }
  }
  return y;
}

function layerNorm(x, T, d, g, b) {
  const y = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    let mu = 0, v = 0;
    for (let i = 0; i < d; i++) mu += x[t * d + i];
    mu /= d;
    for (let i = 0; i < d; i++) v += (x[t * d + i] - mu) ** 2;
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[t * d + i] = (x[t * d + i] - mu) * inv * g[i] + b[i];
  }
  return y;
}

function block(W, p, x, T, d, heads, pad) {
  const hd = d / heads;
  const qkv = linear(layerNorm(x, T, d, W[p + "ln1.weight"], W[p + "ln1.bias"]), T, d, W[p + "qkv.weight"], W[p + "qkv.bias"], 3 * d);
  const out = new Float32Array(T * d);
  const sc = new Float32Array(T);
  for (let h = 0; h < heads; h++) {
    for (let i = 0; i < T; i++) {
      let mx = -Infinity;
      for (let j = 0; j < T; j++) {
        let s = 0;
        for (let e = 0; e < hd; e++) s += qkv[i * 3 * d + h * hd + e] * qkv[j * 3 * d + d + h * hd + e];
        s = pad[j] ? -1e9 : s / Math.sqrt(hd);
        sc[j] = s;
        if (s > mx) mx = s;
      }
      let z = 0;
      for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
      for (let e = 0; e < hd; e++) {
        let s = 0;
        for (let j = 0; j < T; j++) s += sc[j] * qkv[j * 3 * d + 2 * d + h * hd + e];
        out[i * d + h * hd + e] = s / z;
      }
    }
  }
  const a = linear(out, T, d, W[p + "proj.weight"], W[p + "proj.bias"], d);
  for (let i = 0; i < T * d; i++) x[i] += a[i];
  const h1 = linear(layerNorm(x, T, d, W[p + "ln2.weight"], W[p + "ln2.bias"]), T, d, W[p + "fc1.weight"], W[p + "fc1.bias"], 4 * d);
  for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
  const h2 = linear(h1, T, 4 * d, W[p + "fc2.weight"], W[p + "fc2.bias"], d);
  for (let i = 0; i < T * d; i++) x[i] += h2[i];
}

// img: Uint8Array 64*64*3 (HWC), tok: int[max_len], state: [x, y] -> H actions [[ax, ay], ...]
export function predict(model, img, tok, state, heads = 4) {
  const { W, d, meta } = model;
  const H = meta.chunk, L = meta.max_len;
  let S = 64, x = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) x[c * S * S + i] = img[i * 3 + c] / 255 - 0.5;
  const convs = [[32, 5, 2], [64, 3, 1], [d, 3, 1]];
  let C = 3;
  convs.forEach(([O, k, pad], i) => {
    [x, S] = conv(x, C, S, W[`conv.${i}.weight`], W[`conv.${i}.bias`], O, k, 2, pad);
    C = O;
  });
  const P = S * S, T = P + L + 1 + H;
  const seq = new Float32Array(T * d);
  for (let p = 0; p < P; p++) for (let e = 0; e < d; e++) seq[p * d + e] = x[e * P + p] + W.img_pos[p * d + e];
  for (let t = 0; t < L; t++) for (let e = 0; e < d; e++) seq[(P + t) * d + e] = W["word.weight"][tok[t] * d + e] + W.word_pos[t * d + e];
  const st = linear(Float32Array.from(state), 1, 2, W["state.weight"], W["state.bias"], d);
  seq.set(st, (P + L) * d);
  for (let h = 0; h < H; h++) for (let e = 0; e < d; e++) seq[(P + L + 1 + h) * d + e] = W.query[h * d + e];
  const pad = new Uint8Array(T);
  for (let t = 0; t < L; t++) pad[P + t] = tok[t] === 0 ? 1 : 0;
  for (let l = 0; l < model.layers; l++) block(W, `blocks.${l}.`, seq, T, d, heads, pad);
  const q = layerNorm(seq.subarray((T - H) * d), H, d, W["ln.weight"], W["ln.bias"]);
  const y = linear(q, H, d, W["head.weight"], W["head.bias"], 2);
  const out = [];
  for (let h = 0; h < H; h++) out.push([y[2 * h], y[2 * h + 1]]);
  return out;
}
