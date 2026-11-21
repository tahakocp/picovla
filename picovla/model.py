"""The whole policy: CNN patches + word embeddings + fingertip state + action queries,
one transformer over all of them, a linear head on the query tokens.
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
    def __init__(self, d=128, heads=4, layers=4):
        super().__init__()
        self.conv = nn.ModuleList([nn.Conv2d(3, 32, 5, 2, 2), nn.Conv2d(32, 64, 3, 2, 1), nn.Conv2d(64, d, 3, 2, 1)])
        self.img_pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.word = nn.Embedding(len(VOCAB), d)
        self.word_pos = nn.Parameter(torch.randn(MAX_LEN, d) * 0.02)
        self.state = nn.Linear(2, d)
        self.query = nn.Parameter(torch.randn(H, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, img, tok, state):
        """img (B,64,64,3) uint8, tok (B,MAX_LEN) int, state (B,2) -> actions (B,H,2)"""
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
        return self.head(self.ln(x[:, -H:]))
