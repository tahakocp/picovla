"""web/sim.js and web/model.js must agree with the Python reference."""
import json
import os
import shutil
import subprocess
import unittest
import numpy as np
from picovla.sim import Env, render, MAX_STEPS
from picovla.expert import act

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "web", "model")


def run_js(payload):
    p = subprocess.run(["node", os.path.join(ROOT, "tests", "js_runner.mjs")], input=json.dumps(payload),
                       capture_output=True, text=True, check=True, cwd=ROOT)
    return json.loads(p.stdout)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ParityTest(unittest.TestCase):
    def test_sim_bit_exact(self):
        rng = np.random.default_rng(11)
        cases, refs = [], []
        for i in range(40):
            e = Env.random(rng)
            c = dict(blocks={k: list(v) for k, v in e.blocks.items()}, tip=list(e.tip),
                     task=dict(kind=e.task.kind, color=str(e.task.color), arg=None if e.task.arg is None else str(e.task.arg)))
            acts = []
            for t in range(MAX_STEPS // 2):
                a = act(e) if i % 2 else tuple(rng.uniform(-1.2, 1.2, 2))
                acts.append([float(a[0]), float(a[1])])
                e.step(a)
            c["actions"] = acts
            cases.append(c)
            refs.append(e)
        out = run_js({"sim": cases})["sim"]
        bad_px = 0
        for e, o in zip(refs, out):
            self.assertEqual(o["tip"], e.tip)
            for k in e.blocks:
                self.assertEqual(o["blocks"][k], e.blocks[k])
            bad_px += int((np.array(o["img"], np.uint8).reshape(64, 64, 3) != render(e)).any(-1).sum())
        # atan2/acos for the drawn elbow may differ by one ulp between libm and V8
        self.assertLessEqual(bad_px, 4)

    @unittest.skipUnless(os.path.exists(MODEL + ".bin"), "export the model first")
    def test_model_matches_torch(self):
        import torch
        from picovla.model import PicoVLA
        from picovla.lang import tokenize
        from picovla.export import load_web
        m = load_web(PicoVLA(), MODEL)
        m.eval()
        rng = np.random.default_rng(5)
        cases, texts = [], ["push the red block left", "touch blue", "bring the green one to the yellow one", "nudge red up"]
        for i in range(4):
            e = Env.random(rng)
            cases.append(dict(img=render(e).ravel().tolist(), text=texts[i], state=[float(v) for v in e.tip]))
        js = run_js({"sim": [], "model": {"path": os.path.relpath(MODEL, ROOT), "cases": cases}})["model"]
        for c, o in zip(cases, js):
            self.assertEqual(o["tok"], tokenize(c["text"]))
            with torch.no_grad():
                ref = m(torch.tensor(c["img"], dtype=torch.uint8).view(1, 64, 64, 3), torch.tensor([o["tok"]]),
                        torch.tensor([c["state"]])).numpy()[0]
            np.testing.assert_allclose(np.array(o["act"]), ref, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
