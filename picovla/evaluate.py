"""Closed-loop success rate on scenes the policy has never seen.

Every scene, task and phrase comes from a fixed seed, so the numbers in the
README are reproducible with `python -m picovla.evaluate`."""
import argparse
import json
import numpy as np
import torch
from .sim import Env, Task, render, MAX_STEPS, DIRS
from .lang import phrase, tokenize
from .expert import act
from .model import PicoVLA

KINDS = ["touch", "push", "toward"]


def scenes(kind, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        e = Env.random(rng)
        if e.task.kind == kind:
            out.append(e)
    return out


def other_task(env, rng):
    """Same kind of task, different target colour: tests whether the policy reads the words."""
    t = env.task
    c = [x for x in env.blocks if x != t.color and x != t.arg][0]
    if t.kind == "toward":
        return Task("toward", c, t.arg if t.arg != c else t.color)
    return Task(t.kind, c, t.arg)


@torch.no_grad()
def run(model, envs, texts, execute=4):
    tok = torch.tensor([tokenize(s) for s in texts])
    done = np.zeros(len(envs), bool)
    for t in range(MAX_STEPS):
        if t % execute == 0:
            live = np.flatnonzero(~done)
            if len(live) == 0:
                break
            img = torch.from_numpy(np.stack([render(envs[i]) for i in live]))
            st = torch.from_numpy(np.stack([envs[i].state() for i in live]))
            plan = dict(zip(live, model(img, tok[live], st).numpy()))
        for i in np.flatnonzero(~done):
            done[i] = envs[i].step(plan[i][t % execute])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="web/model", help="a .pt checkpoint or the web/model weights")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--execute", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--expert", action="store_true", help="also score the scripted expert on the same scenes")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = PicoVLA()
    if a.ckpt.endswith(".pt"):
        model.load_state_dict(torch.load(a.ckpt))
    else:
        from .export import load_web
        load_web(model, a.ckpt)
    model.eval()
    res = {}
    for k, kind in enumerate(KINDS):
        rng = np.random.default_rng(100 + k)
        for split in ["train", "heldout"]:
            envs = scenes(kind, a.n, 10_000 + k)
            res[f"{kind}/{split}"] = float(run(model, envs, [phrase(e.task, rng, split) for e in envs], a.execute).mean())
        if a.expert:
            envs = scenes(kind, a.n, 10_000 + k)
            res[f"{kind}/expert"] = float(np.mean([any(e.step(act(e)) for _ in range(MAX_STEPS)) for e in envs]))
        if a.ablate:
            envs = scenes(kind, a.n, 10_000 + k)
            other = [other_task(e, rng) for e in envs]
            wrong = [phrase(t, rng) for t in other]
            res[f"{kind}/wrong-colour"] = float(run(model, envs, wrong, a.execute).mean())
            # same scenes and words, but scored against the instruction it was given
            envs = scenes(kind, a.n, 10_000 + k)
            for e, t in zip(envs, other):
                e.task = t
            res[f"{kind}/swapped-followed"] = float(run(model, envs, wrong, a.execute).mean())
        print({x: y for x, y in res.items() if x.startswith(kind)}, flush=True)
    res["all/train"] = float(np.mean([res[f"{k}/train"] for k in KINDS]))
    res["all/heldout"] = float(np.mean([res[f"{k}/heldout"] for k in KINDS]))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
