import unittest
import numpy as np
from picovla.sim import Env, Task, render, MAX_STEPS, RB, RT
from picovla.expert import act
from picovla.lang import tokenize, phrase, VOCAB, TRAIN, HELDOUT


class SimTest(unittest.TestCase):
    def test_expert_solves_almost_everything(self):
        rng = np.random.default_rng(7)
        ok = []
        for _ in range(500):
            e = Env.random(rng)
            ok.append(any(e.step(act(e)) for _ in range(MAX_STEPS)))
        self.assertGreaterEqual(np.mean(ok), 0.99)

    def test_pushing_moves_the_block(self):
        e = Env({"red": (0.0, 0.0), "blue": (0.5, 0.4), "green": (-0.5, 0.4)}, (0.0, -0.3), Task("push", "red", "up"))
        for _ in range(30):
            e.step((0.0, 1.0))
        self.assertGreater(e.blocks["red"][1], 0.5)
        self.assertTrue(e.success())

    def test_no_overlap_after_steps(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            e = Env.random(rng)
            for _ in range(60):
                e.step(rng.uniform(-1, 1, 2))
                for b in e.blocks.values():
                    d = np.hypot(b[0] - e.tip[0], b[1] - e.tip[1])
                    self.assertGreater(d, RB + RT - 1e-3)

    def test_wrong_block_is_not_success(self):
        e = Env({"red": (0.0, 0.0), "blue": (0.5, 0.0), "green": (-0.5, 0.4)}, (0.5, -0.2), Task("touch", "red"))
        for _ in range(20):
            e.step((0.0, 1.0))
        self.assertFalse(e.success())

    def test_render_shape_and_colours(self):
        e = Env({"red": (0.0, 0.0), "blue": (0.5, 0.4), "green": (-0.5, 0.4)}, (0.0, -0.5), Task("touch", "red"))
        img = render(e)
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertEqual(tuple(img[32, 32]), (220, 50, 47))


class LangTest(unittest.TestCase):
    def test_heldout_phrases_are_new(self):
        for k in TRAIN:
            self.assertFalse(set(TRAIN[k]) & set(HELDOUT[k]))

    def test_tokenize(self):
        t = tokenize("Push the RED block left!")
        self.assertEqual(len(t), 10)
        self.assertEqual([VOCAB[i] for i in t[:5]], ["push", "the", "red", "block", "left"])
        self.assertEqual(tokenize("nudge red")[0], 1)

    def test_phrase_mentions_colour(self):
        rng = np.random.default_rng(0)
        for split in ("train", "heldout"):
            self.assertIn("green", phrase(Task("toward", "green", "blue"), rng, split))


if __name__ == "__main__":
    unittest.main()
