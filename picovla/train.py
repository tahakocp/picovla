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
    ap.add_argument("--head", choices=["l1", "flow"], default="l1")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--init", default=None, help="start the backbone from these weights (.pt or web/model)")
    ap.add_argument("--freeze", action="store_true", help="train only the action head")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.data.endswith(".npz"):
        d = {k: torch.from_numpy(v) for k, v in np.load(a.data).items()}
    else:   # a directory of .npy files: memory-mapped, so two runs can share the 1.2 GB of frames
        d = {k: np.load(f"{a.data}/{k}.npy", mmap_mode="r") for k in ["img", "tok", "state", "act", "mask"]}
    n = len(d["img"])
    model = PicoVLA(head=a.head)
    if a.init:
        src = PicoVLA()
        if a.init.endswith(".pt"):
            src.load_state_dict(torch.load(a.init))
        else:
            from .export import load_web
            load_web(src, a.init)
        sd = {k: v for k, v in src.state_dict().items() if not k.startswith("head.")}
        model.load_state_dict(sd, strict=False)
    if a.freeze:
        for k, p in model.named_parameters():
            p.requires_grad = k.startswith(("head.", "flow."))
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {n:,} frames")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    g = torch.Generator().manual_seed(a.seed)
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        i = torch.randint(0, n, (a.batch,), generator=g)
        b = {k: v[i] if torch.is_tensor(v) else torch.from_numpy(v[np.sort(i.numpy())]) for k, v in d.items()}
        loss = model.loss(b["img"], b["tok"], b["state"], b["act"], b["mask"])
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
