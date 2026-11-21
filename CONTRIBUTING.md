# Contributing

The point of this repo is that someone can read all of it in an evening. Keep changes small and keep the numbers honest.

- Run `python -m unittest discover -s tests -v` before you open a PR. If you touch `web/`, also run `npm install && N=24 npm run e2e`.
- If you change `sim.py`, change `web/sim.js` in the same way. The parity test compares positions with `==`, so keep the same operation order.
- If you change the model, re-export (`python -m picovla.export`) so the browser demo and the Python code agree, and paste the new `python -m picovla.evaluate --ablate --expert` output into the PR.
- New features (another action head, another task) should add a row to the results table rather than replace the old one.
- No new heavy dependencies in the core. numpy, torch and pillow are the whole list.
