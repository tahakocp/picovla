# picovla

A vision-language-action robot policy written from scratch in about 530 lines of PyTorch. It looks at a 64x64 picture, reads an English instruction and moves a two-link arm, with either an L1 or a flow-matching action head. It trains in 40 minutes on a CPU, and the same weights run in your browser in plain JavaScript.

**[Try it in the browser](https://tahakocp.github.io/picovla/)** (3.6 MB, no install, nothing leaves the page)

![four unseen scenes, one instruction each](docs/demo.gif)

None of these scenes were in the training data, and "shove the green block up" is a phrasing it never saw. The bottom-right one fails, which is close to the real rate: the table below has the numbers.

```bash
git clone https://github.com/tahakocp/picovla && cd picovla
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # CPU torch is enough
python -m picovla.evaluate --ablate      # score the shipped weights, ~30 s
```

## What it does, measured

200 fixed random scenes per task, none of them in the training data. Success means the task check in `sim.py` passed within 120 steps.

| task | example | trained phrasing | unseen phrasing | scripted expert |
|---|---|---|---|---|
| touch | "touch the red block" | 100% | 97.5% | 100% |
| push | "push the blue block left" | 78.0% | 79.0% | 100% |
| toward | "move green next to yellow" | 58.5% | 50.0% | 100% |
| all | | 78.8% | 75.5% | 100% |

Does it actually read the words? Same 200 scenes per task, but the instruction names a different block:

| task | still does the original task | does what the new words say |
|---|---|---|
| touch | 30.5% | 100% |
| push | 0.0% | 79.5% |
| toward | 4.0% | 61.0% |

So the instruction changes the behaviour, and the policy follows the new one about as well as the original. Touch counts as soon as the fingertip comes close, so brushing past the original block on the way to the new one also scores; that is where the 30.5% comes from.

![same scene, three instructions](docs/same-scene.gif)

Numbers come from `docs/eval.json`, produced by `python -m picovla.evaluate --ablate --expert --out docs/eval.json`. Every scene, task and phrase comes from a fixed seed, so you should get the same table.

| | |
|---|---|
| parameters | 902,210 (L1 head), 1,005,890 (flow head) |
| demonstrations | 3,000 episodes, 96,552 frames, collected in 27 s |
| training | 12,000 steps, batch 128, 2,388 s on an 8-vCPU Xeon cloud VM, no GPU |
| browser | 147 ms median per forward pass in headless Chrome on the same VM, one pass per 4 steps |
| Python core | 532 non-blank, non-comment lines in `picovla/` (466 before the flow head) |
| JS forward pass | 124 lines in `web/model.js`, no ML library |

## Two action heads, same backbone

New in 0.2: a flow-matching head (as in pi0) next to the original L1 head (as in ACT). Pick one with `--head flow`. It is a 3-layer MLP that turns noise into the 8x2 action chunk in a few Euler steps; the transformer still runs once per chunk, so only the MLP repeats.

Success on the same 200 unseen scenes per task as above:

| head | training | trained phrasing | unseen phrasing | "next to", unseen | CPU per chunk |
|---|---|---|---|---|---|
| L1 (shipped) | 12,000 steps | 78.8% | 75.5% | 50.0% | 2.29 ms |
| flow, 10 steps | L1 backbone frozen, 1,500 head-only steps (102 s) | 79.2% | 76.7% | 55.5% | 3.05 ms |
| flow, 5 steps | same weights | 80.8% | 75.8% | 50.5% | 2.67 ms |
| flow, 1 step | same weights | 81.0% | 80.2% | 62.0% | 2.41 ms |
| L1 | from scratch, 2,400 steps | 20.3% | 30.0% | 13.0% | |
| flow, 10 steps | from scratch, 2,400 steps | 9.2% | 11.8% | 1.0% | |

What I take from it:

- On a trained backbone the flow head is at least as good as L1. The 1-step row is the best, but with 200 scenes per task one standard error is about 3 points per task, so only the "next to" gain (50% to 62%) is clearly more than noise.
- Trained from scratch on a short budget, flow is behind: it learns slower than L1 here. I only had CPU time for 2,400 steps per head, so this says nothing about where it ends up after 12,000.
- The expert is deterministic, so there is no multimodal action distribution for flow matching to model. The benefit pi0 reports comes from data with several valid ways to do a task, which this toy does not have.

CPU per chunk is batch 1, one thread, median of 50 (`python scripts/bench_heads.py`). All rows are in `docs/heads.json`. The browser demo still runs the L1 head.

## How it works

Read the files in this order.

1. [`picovla/sim.py`](picovla/sim.py). A top-down table with three coloured discs and a two-link arm. Only the fingertip collides. The physics uses `+ - * /` and `sqrt` and nothing else, so `web/sim.js` reproduces it bit for bit.
2. [`picovla/expert.py`](picovla/expert.py). A scripted policy that solves every task: go behind the block, line up, push through. It only looks at the current state, so the cloned policy never sees two answers for one picture.
3. [`picovla/lang.py`](picovla/lang.py). 11 phrasing templates for training, 6 held out for testing, a 23-word vocabulary. No pretrained text model. The word embeddings start random like everything else.
4. [`picovla/data.py`](picovla/data.py). Runs the expert and records (image, words, fingertip, next 8 actions). Half the episodes add noise to the executed action but keep the clean label (DART). Without this the policy never learns to recover from its own small mistakes.
5. [`picovla/model.py`](picovla/model.py). Three strided convolutions turn the image into 64 patch tokens. Add 10 word tokens, one fingertip token and 8 learned action queries, run a 4-layer transformer over all 83 tokens, read 8 (dx, dy) actions off the query tokens with a linear layer, or sample them with the flow-matching head. The attention is written out by hand so the JS port can follow it line by line.
6. [`picovla/train.py`](picovla/train.py). L1 loss on the action chunk as in ACT, or the rectified-flow velocity loss with `--head flow`. AdamW, one-cycle schedule.
7. [`picovla/evaluate.py`](picovla/evaluate.py). Closed loop: predict 8 actions, execute 4, look again.
8. [`picovla/export.py`](picovla/export.py) and [`web/model.js`](web/model.js). Weights to one float32 file, then the same forward pass in JavaScript.

This is the shape of real VLAs (OpenVLA, pi0, SmolVLA) with every part made as small as it can be: a vision encoder, a language encoder, a state input, one transformer, an action-chunk head. Real ones start from pretrained vision and language models, train on hours of robot data and predict 7-DoF actions. The toy keeps the plumbing and drops the scale.

## Train your own

```bash
python -m picovla.data --episodes 3000        # 1.2 GB in data/demos.npz
python -m picovla.train --steps 12000         # checkpoints/picovla.pt, ~40 min on 8 CPU cores
python -m picovla.evaluate --ckpt checkpoints/picovla.pt --ablate
python -m picovla.export                      # writes web/model.bin + web/model.json

# flow head on top of the shipped backbone, ~2 min on 8 cores
python -m picovla.train --head flow --init web/model --freeze --steps 1500 --out checkpoints/flow.pt
python -m picovla.evaluate --head flow --flow-steps 1 --ckpt checkpoints/flow.pt
npm run serve                                 # http://127.0.0.1:8000
```

The trained L1 checkpoint is attached to the [v0.1.0 release](https://github.com/tahakocp/picovla/releases/tag/v0.1.0), the flow checkpoint to [v0.2.0](https://github.com/tahakocp/picovla/releases/tag/v0.2.0).

## Tests

```bash
python -m unittest discover -s tests -v     # 12 tests
N=24 npm run e2e                            # needs Chrome; drives the real page
```

- The scripted expert solves at least 99% of 500 random scenes (100% of the 600 evaluation scenes).
- `web/sim.js` matches `sim.py` exactly (block and tip positions compared with `==`) over 40 episodes of 60 steps, half random actions, half expert. The drawn elbow may differ by one pixel because `atan2` and `acos` can differ by one ulp between libm and V8; the test allows 4 pixels in 40 frames.
- `web/model.js` matches the PyTorch forward pass within 2e-4 on four scenes, including an instruction with an unknown word.
- The e2e test loads the page in headless Chrome and runs instructions through the UI. Last local run: 18/24 done.

## Limits

- It is a 2D toy. Discs, a fingertip, no grasping, no friction model, no camera noise. Nothing here was run on a real robot.
- "toward" (push one block next to another) works about half the time. The expert needs to go round the block and the cloned policy often stops short or drifts past. More data did not fix it: a second run on 6,000 episodes for 20,000 steps scored 39% overall, so I kept the first one.
- Language understanding is word-level. It knows 21 words. Unknown words become `<unk>` and are ignored, which is why "shove" and "poke" still work: the colour and direction carry the meaning.
- Only one seed was trained for the shipped weights and for each head in the comparison.
- All timings are CPU only. Nothing was trained or timed on a GPU.

## Next to the real ones

| | picovla | [SmolVLA](https://huggingface.co/blog/smolvla) | [pi0](https://github.com/Physical-Intelligence/openpi) | [OpenVLA](https://github.com/openvla/openvla) |
|---|---|---|---|---|
| parameters | 0.9M (1.0M with the flow head) | ~450M | ~3.3B | 7B |
| action head | L1 chunk or flow matching | flow matching | flow matching | discrete tokens |
| pretrained parts | none | VLM | VLM | VLM |
| world | 2D toy sim | real arms (SO-100) | real arms | real arms |
| runs on | a laptop CPU, a browser tab | one GPU | GPU | GPU |

picovla is for reading the whole loop in an afternoon, not for controlling a robot. If you want a real one, [LeRobot](https://github.com/huggingface/lerobot) has SmolVLA, pi0 and ACT with training and hardware support. Their numbers come from their papers; picovla's are from this repo.

## Ideas for contributions

- The flow head in the browser (web/model.js), and a from-scratch flow run for the full 12,000 steps.
- Demonstrations with two valid ways to do a task, where flow matching should beat L1.
- A wrist-camera view or a second camera.
- Swap the word embeddings for a small frozen text encoder and test phrasing it has never seen.
- A 3D version on MuJoCo with the same file layout.

See [CONTRIBUTING.md](CONTRIBUTING.md). If you add a feature, add its row to the results table.

## Credits

The policy follows ACT ([tonyzhaozh/act](https://github.com/tonyzhaozh/act)): action chunks, L1 loss, sim / scripted expert / record / train / evaluate split. DART noise injection is from Laskey et al., 2017. The flat, readable layout is inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). [keivalya/mini-vla](https://github.com/keivalya/mini-vla) is another small VLA worth reading, with a diffusion head on a MuJoCo task.

Every number above was measured on the machine described in the table.

MIT license.
