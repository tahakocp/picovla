"""Median CPU time for one action chunk (batch 1, one thread) for each head."""
import os
import sys
import time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from picovla.model import PicoVLA

torch.set_num_threads(1)
img = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
tok, st = torch.randint(1, 20, (1, 10)), torch.zeros(1, 2)
for name, kw in [("l1", {}), ("flow, 1 step", dict(head="flow", flow_steps=1)),
                 ("flow, 5 steps", dict(head="flow", flow_steps=5)), ("flow, 10 steps", dict(head="flow", flow_steps=10))]:
    m = PicoVLA(**kw).eval()
    ts = []
    with torch.no_grad():
        for i in range(60):
            t0 = time.perf_counter()
            m(img, tok, st)
            ts.append(time.perf_counter() - t0)
    ts = sorted(ts[10:])
    print(f"{name:15s} {sum(p.numel() for p in m.parameters()):>9,} params  {ts[len(ts) // 2] * 1e3:.2f} ms")
