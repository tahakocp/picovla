"""docs/demo.gif: four unseen scenes, the trained policy, the instruction it got."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from picovla.sim import Env, Task, render, MAX_STEPS
from picovla.lang import tokenize
from picovla.model import PicoVLA

CASES = [  # (seed, task, instruction) picked before looking at results; seeds are scene seeds
    (1, Task("push", None, "left"), "push the {c} block left"),
    (2, Task("toward", None, None), "move {c} next to {o}"),
    (3, Task("touch", None), "reach the {c} block"),
    (4, Task("push", None, "up"), "shove the {c} block up"),
]
S = 288


def font(sz):
    for f in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    m = PicoVLA()
    m.load_state_dict(torch.load("checkpoints/picovla.pt"))
    m.eval()
    runs = []
    for seed, task, text in CASES:
        e = Env.random(np.random.default_rng(seed))
        cs = sorted(e.blocks)
        task.color = cs[0]
        if task.kind == "toward":
            task.arg = cs[1]
        e.task = task
        text = text.format(c=task.color, o=task.arg)
        frames, plan, done = [render(e, S)], None, False
        tok = torch.tensor([tokenize(text)])
        for t in range(MAX_STEPS):
            if t % 4 == 0:
                with torch.no_grad():
                    plan = m(torch.from_numpy(render(e))[None], tok, torch.from_numpy(e.state())[None])[0].numpy()
            done = e.step(plan[t % 4])
            frames.append(render(e, S))
            if done:
                break
        runs.append((text, frames, done))
    n = max(len(f) for _, f, _ in runs) + 8
    f1, f2 = font(17), font(15)
    out = []
    for i in range(0, n, 2):
        canvas = Image.new("RGB", (2 * S + 12, 2 * (S + 30) + 6), (250, 248, 243))
        d = ImageDraw.Draw(canvas)
        for k, (text, frames, done) in enumerate(runs):
            x, y = (k % 2) * (S + 6) + 3, (k // 2) * (S + 32) + 3
            canvas.paste(Image.fromarray(frames[min(i, len(frames) - 1)]), (x, y + 28))
            d.text((x + 4, y + 5), f'"{text}"', fill=(30, 30, 30), font=f1)
            if i >= len(frames) - 1:
                if done:
                    d.text((x + S - 52, y + 36), "done", fill=(26, 127, 55), font=f2)
                else:
                    d.text((x + S - 128, y + 36), "failed, 120 steps", fill=(190, 30, 30), font=f2)
        out.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    out[0].save("docs/demo.gif", save_all=True, append_images=out[1:], duration=70, loop=0, optimize=True)
    print("docs/demo.gif", len(out), "frames", [(t, len(f) - 1, ok) for t, f, ok in runs])


if __name__ == "__main__":
    main()
