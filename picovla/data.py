"""Collect expert demonstrations.

Half the episodes execute the expert action plus Gaussian noise but store the
clean expert action as the label (DART, Laskey et al. 2017). Without it the
cloned policy never sees how to recover from its own small mistakes.
"""
import argparse
import numpy as np
from .sim import Env, render, MAX_STEPS
from .expert import act
from .lang import phrase, tokenize

H = 8  # action chunk length


def collect(n_episodes, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    imgs, toks, states, chunks, masks = [], [], [], [], []
    for ep in range(n_episodes):
        env = Env.random(rng)
        tok = tokenize(phrase(env.task, rng))
        sigma = noise if ep % 2 else 0.0
        ep_imgs, ep_states, ep_acts = [], [], []
        for _ in range(MAX_STEPS):
            a = act(env)
            ep_imgs.append(render(env))
            ep_states.append(env.state())
            ep_acts.append(a)
            done = env.step(np.asarray(a) + rng.normal(0, sigma, 2) if sigma else a)
            if done:
                break
        acts = np.asarray(ep_acts, np.float32)
        T = len(acts)
        for t in range(T):
            c = np.zeros((H, 2), np.float32)
            m = np.zeros(H, np.float32)
            k = min(H, T - t)
            c[:k], m[:k] = acts[t:t + k], 1.0
            chunks.append(c)
            masks.append(m)
        imgs += ep_imgs
        states += ep_states
        toks += [tok] * T
    return dict(img=np.stack(imgs), tok=np.asarray(toks, np.int64), state=np.stack(states),
                act=np.stack(chunks), mask=np.stack(masks))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/demos.npz")
    a = ap.parse_args()
    d = collect(a.episodes, a.seed)
    np.savez(a.out, **d)
    print(f"{a.episodes} episodes, {len(d['img'])} frames -> {a.out}")
