import argparse
import time
import numpy as np
import torch
from .model import PicoVLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demos.npz")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/picovla.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    d = {k: torch.from_numpy(v) for k, v in np.load(a.data).items()}
    n = len(d["img"])
    model = PicoVLA()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {n:,} frames")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    g = torch.Generator().manual_seed(a.seed)
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        i = torch.randint(0, n, (a.batch,), generator=g)
        pred = model(d["img"][i], d["tok"][i], d["state"][i])
        m = d["mask"][i][..., None]
        loss = ((pred - d["act"][i]).abs() * m).sum() / (m.sum() * 2)   # L1, as in ACT
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 3000 == 0:
            torch.save(model.state_dict(), a.out)
    torch.save(model.state_dict(), a.out)
    print(f"saved {a.out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
