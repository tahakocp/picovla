"""Write the weights as one float32 blob + a JSON index for web/model.js."""
import argparse
import json
import numpy as np
import torch
from .lang import VOCAB, MAX_LEN
from .data import H


def load_web(model, base="web/model"):
    """Load the shipped web weights back into a PicoVLA (used by tests and evaluate)."""
    meta = json.load(open(base + ".json"))
    flat = np.fromfile(base + ".bin", dtype="<f4")
    sd = {}
    for k, t in meta["tensors"].items():
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        sd[k] = torch.from_numpy(flat[t["offset"]:t["offset"] + n].copy()).view(t["shape"])
    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/picovla.pt")
    ap.add_argument("--out", default="web/model")
    a = ap.parse_args()
    sd = torch.load(a.ckpt)
    index, blobs, off = {}, [], 0
    for k, v in sd.items():
        arr = v.float().numpy().astype("<f4").ravel()
        index[k] = {"shape": list(v.shape), "offset": off}
        blobs.append(arr)
        off += arr.size
    np.concatenate(blobs).tofile(a.out + ".bin")
    json.dump({"vocab": VOCAB, "max_len": MAX_LEN, "chunk": H, "tensors": index}, open(a.out + ".json", "w"))
    print(f"{off:,} floats -> {a.out}.bin ({off * 4 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
