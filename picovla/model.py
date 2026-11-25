"""The whole policy: CNN patches + word embeddings + fingertip state + action queries,
one transformer over all of them, then an action head on the query tokens:
either a linear L1 head (as in ACT) or a small flow-matching head (as in pi0).
Written out by hand (no nn.TransformerEncoder) so web/model.js can mirror it line by line."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lang import VOCAB, MAX_LEN
from .data import H


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x, pad):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.heads)
        att = att.masked_fill(pad[:, None, None, :], -1e9).softmax(-1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class PicoVLA(nn.Module):
    def __init__(self, d=128, heads=4, layers=4, head="l1", flow_steps=10):
        super().__init__()
        self.kind, self.flow_steps = head, flow_steps
        self.conv = nn.ModuleList([nn.Conv2d(3, 32, 5, 2, 2), nn.Conv2d(32, 64, 3, 2, 1), nn.Conv2d(64, d, 3, 2, 1)])
        self.img_pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.word = nn.Embedding(len(VOCAB), d)
        self.word_pos = nn.Parameter(torch.randn(MAX_LEN, d) * 0.02)
        self.state = nn.Linear(2, d)
        self.query = nn.Parameter(torch.randn(H, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        if head == "l1":
            self.head = nn.Linear(d, 2)
        else:
            self.flow = FlowHead(d)

    def forward(self, img, tok, state, noise=None):
        """img (B,64,64,3) uint8, tok (B,MAX_LEN) int, state (B,2) -> actions (B,H,2)"""
        h = self.features(img, tok, state)
        if self.kind == "l1":
            return self.head(h)
        return self.flow.sample(h, self.flow_steps, noise)

    def loss(self, img, tok, state, act, mask):
        h = self.features(img, tok, state)
        m = mask[..., None]
        if self.kind == "l1":
            return ((self.head(h) - act).abs() * m).sum() / (m.sum() * 2)   # L1, as in ACT
        return self.flow.loss(h, act, m)

    def features(self, img, tok, state):
        x = img.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        for c in self.conv:
            x = F.gelu(c(x), approximate="tanh")
        vis = x.flatten(2).transpose(1, 2) + self.img_pos           # (B,64,d)
        txt = self.word(tok) + self.word_pos                         # (B,L,d)
        st = self.state(state)[:, None]                              # (B,1,d)
        q = self.query.expand(img.shape[0], -1, -1)                  # (B,H,d)
        x = torch.cat([vis, txt, st, q], 1)
        pad = torch.cat([torch.zeros(img.shape[0], 64, dtype=torch.bool, device=img.device), tok == 0,
                         torch.zeros(img.shape[0], 1 + H, dtype=torch.bool, device=img.device)], 1)
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x[:, -H:])


class FlowHead(nn.Module):
    """Rectified flow over the 8x2 action chunk. The transformer runs once per chunk;
    only this 3-layer MLP runs at each of the integration steps, so sampling costs
    about 0.4 MFLOP per step on top of the ~60 MFLOP backbone."""

    def __init__(self, d, hidden=256):
        super().__init__()
        self.inp = nn.Linear(d + 2 + 16, hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)

    def velocity(self, h, x, t):
        f = t[..., None] * torch.arange(1, 9, device=h.device) * math.pi         # (B,1,8)
        te = torch.cat([f.sin(), f.cos()], -1).expand(-1, h.shape[1], -1)
        z = F.gelu(self.inp(torch.cat([h, x, te], -1)), approximate="tanh")
        return self.out(F.gelu(self.mid(z), approximate="tanh"))

    def loss(self, h, act, m):
        e = torch.randn_like(act)
        t = torch.rand(act.shape[0], 1, device=act.device)
        x = t[..., None] * act + (1 - t[..., None]) * e                          # t=0 noise, t=1 data
        return (((self.velocity(h, x, t) - (act - e)) ** 2) * m).sum() / (m.sum() * 2)

    def sample(self, h, steps, noise=None):
        x = torch.randn(h.shape[0], h.shape[1], 2) if noise is None else noise
        for i in range(steps):
            t = torch.full((h.shape[0], 1), i / steps)
            x = x + self.velocity(h, x, t) / steps
        return x
