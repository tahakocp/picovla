# Changelog

## 0.2.0 (2026-11-25)

- Flow-matching action head (`--head flow`, `--flow-steps`): a 3-layer MLP that integrates a rectified flow over the 8x2 chunk; the transformer runs once per chunk. 1,005,890 parameters with the flow head.
- `train.py --init web/model --freeze` trains only the head on top of existing weights.
- Head comparison on the 600 evaluation scenes (`docs/heads.json`): flow on the frozen L1 backbone 81.0% / 80.2% (1 step) vs L1 78.8% / 75.5%; from scratch at 2,400 steps L1 20.3% vs flow 9.2%.
- `scripts/bench_heads.py`: CPU time per chunk, L1 2.29 ms, flow 2.41 ms (1 step) to 3.05 ms (10 steps).
- Tests for the flow head; CI smoke-trains both heads. Pages workflow split into build and deploy jobs with timeouts.
- The browser demo is unchanged and still runs the L1 head.

## 0.1.0 (2026-11-21)

- First release: 2D tabletop sim, scripted expert with DART data, 0.9M-parameter policy with an L1 chunk head, closed-loop evaluation, browser demo with a JS port of the sim and the forward pass.
