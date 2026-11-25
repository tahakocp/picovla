"""The flow-matching head: shapes, determinism with fixed noise, and that it can fit a chunk."""
import unittest
import torch
from picovla.model import PicoVLA, FlowHead
from picovla.data import H


class FlowHeadTest(unittest.TestCase):
    def batch(self, b=4):
        g = torch.Generator().manual_seed(0)
        img = torch.randint(0, 256, (b, 64, 64, 3), generator=g, dtype=torch.uint8)
        tok = torch.randint(1, 20, (b, 10), generator=g)
        return img, tok, torch.rand(b, 2, generator=g)

    def test_shapes_and_fixed_noise(self):
        torch.manual_seed(0)
        m = PicoVLA(head="flow").eval()
        img, tok, st = self.batch()
        z = torch.randn(4, H, 2)
        with torch.no_grad():
            a, b = m(img, tok, st, z), m(img, tok, st, z.clone())
        self.assertEqual(tuple(a.shape), (4, H, 2))
        self.assertTrue(torch.equal(a, b))

    def test_fits_one_chunk(self):
        # one fixed context, one target chunk: sampling from noise must land near it
        torch.manual_seed(0)
        head = FlowHead(16, hidden=64)
        h = torch.randn(1, H, 16).expand(64, -1, -1)
        act = torch.linspace(-1, 1, H * 2).view(1, H, 2).expand(64, -1, -1)
        opt = torch.optim.Adam(head.parameters(), 3e-3)
        for _ in range(300):
            loss = head.loss(h, act, torch.ones(64, H, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = head.sample(h[:16], 10)
        self.assertLess((out - act[:16]).abs().mean().item(), 0.1)

if __name__ == "__main__":
    unittest.main()
