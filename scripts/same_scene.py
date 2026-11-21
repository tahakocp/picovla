"""docs/same-scene.gif: one scene, three instructions. Only the words change."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image, ImageDraw
from picovla.sim import Env, Task, render, MAX_STEPS
from picovla.lang import tokenize
from picovla.model import PicoVLA
from make_gif import font

SEED = 5  # fixed before the first run
S = 256


def main():
    m = PicoVLA()
    m.load_state_dict(torch.load("checkpoints/picovla.pt"))
    m.eval()
    base = Env.random(np.random.default_rng(SEED))
    c = sorted(base.blocks)
    cases = [(Task("touch", c[0]), f"touch the {c[0]} block"),
             (Task("push", c[1], "right"), f"push the {c[1]} block right"),
             (Task("toward", c[2], c[0]), f"move {c[2]} next to {c[0]}")]
    runs = []
    for task, text in cases:
        e = Env(base.start, base.tip, task)
        tok = torch.tensor([tokenize(text)])
        frames, done = [render(e, S)], False
        for t in range(MAX_STEPS):
            if t % 4 == 0:
                with torch.no_grad():
                    plan = m(torch.from_numpy(render(e))[None], tok, torch.from_numpy(e.state())[None])[0].numpy()
            done = e.step(plan[t % 4])
            frames.append(render(e, S))
            if done:
                break
        runs.append((text, frames, done))
    n = max(len(f) for _, f, _ in runs) + 10
    f1, f2 = font(15), font(14)
    out = []
    for i in range(0, n, 2):
        cv = Image.new("RGB", (3 * S + 16, S + 34), (250, 248, 243))
        d = ImageDraw.Draw(cv)
        for k, (text, frames, done) in enumerate(runs):
            x = k * (S + 6) + 3
            cv.paste(Image.fromarray(frames[min(i, len(frames) - 1)]), (x, 30))
            d.text((x + 3, 7), f'"{text}"', fill=(30, 30, 30), font=f1)
            if i >= len(frames) - 1:
                d.text((x + S - 50, 36), "done" if done else "failed", fill=(26, 127, 55) if done else (190, 30, 30), font=f2)
        out.append(cv.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    out[0].save("docs/same-scene.gif", save_all=True, append_images=out[1:], duration=70, loop=0, optimize=True)
    print("docs/same-scene.gif", [(t, len(f) - 1, ok) for t, f, ok in runs])


if __name__ == "__main__":
    main()
